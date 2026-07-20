from matplotlib.backends.backend_agg import FigureCanvasAgg
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from transformers import PretrainedConfig
from torch.utils.data import Dataset
import matplotlib.pyplot as plt 
import imageio.v2 as imageio
from torch import nn 
import numpy as np 
import textwrap 
import pickle 
import torch 
import copy 
import os 
import matplotlib.pyplot as plt

def prepare_images_for_saving(images_tensor, resolution, grid_size=4, range_type="neg1pos1"):
    if range_type != "uint8":
        images_tensor = (images_tensor * 0.5 + 0.5).clamp(0, 1) * 255

    images = images_tensor[:grid_size*grid_size].permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")
    grid = images.reshape(grid_size, grid_size, resolution, resolution, 3)
    grid = np.swapaxes(grid, 1, 2).reshape(grid_size*resolution, grid_size*resolution, 3)
    return grid

def prepare_debug_output(tensor, resolution):
    # N x T x 3 x H x W 
    N, T = tensor.shape[:2]
    tensor = tensor.transpose(0, 1)
    tensor = ((tensor * 0.5 + 0.5).clamp(0, 1) * 255).permute(0, 1, 3, 4, 2).detach().cpu().numpy().astype("uint8")      
    tensor = np.swapaxes(tensor, 1, 2).reshape(T*resolution, N*resolution, 3)
    return tensor 

def draw_valued_array(data, output_dir, grid_size=4):
    fig = plt.figure(figsize=(20,20))

    data = data[:grid_size*grid_size].reshape(grid_size, grid_size)
    cax = plt.matshow(data, cmap='viridis')  # Change cmap to your desired color map
    plt.colorbar(cax)

    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', color='black')

    plt.savefig(os.path.join(output_dir, "cache.jpg"))
    plt.close('all')

    # read the image 
    image = imageio.imread(os.path.join(output_dir, "cache.jpg"))
    return image

def draw_probability_histogram(data):
    fig = plt.figure(figsize=(5,5))

    plt.hist(data, color='blue', edgecolor='black')
    plt.title('Histogram of Realism Prediction')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Get the canvas as a PIL image
    image = Image.frombytes(
        "RGB", canvas.get_width_height(), canvas.tostring_rgb()
    )
    plt.close('all')
    return image

def draw_gradient_norm(data, pred_realism, num_bin=10, bin_size=0.1):
    mean_list = [] 
    for bin_idx in range(num_bin):
        start = bin_idx * bin_size
        end = (bin_idx + 1) * bin_size
        data_bin = data[(pred_realism >= start) & (pred_realism < end)]

        if len(data_bin) == 0:
            mean_list.append(0)
        else:
            mean_list.append(data_bin.mean())
        
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(num_bin) * bin_size, mean_list)
    plt.title('Gradient Norm')
    plt.xlabel('Predicted Realism')
    plt.ylabel('Mean Grad Norm')

    plt.xlim(0, 1)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Get the canvas as a PIL image
    image = Image.frombytes(
        "RGB", canvas.get_width_height(), canvas.tostring_rgb()
    )
    plt.close('all')
    return image

def draw_array(indices, values, min_val=None, max_val=None):
    fig = plt.figure(figsize=(5,5))
    plt.plot(indices, values)

    if max_val is None: 
        max_val = max(values[values!= 1.0].max() * 1.1, 0.05)
    
    if min_val is None: 
        min_val = 0 

    plt.ylim(min_val, max_val)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Get the canvas as a PIL image
    image = Image.frombytes(
        "RGB", canvas.get_width_height(), canvas.tostring_rgb()
    )
    plt.close('all')
    return image

def cycle(dl):
    while True:
        for data in dl:
            yield data

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

class EMA(nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay

        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # update the parameters
        update_ema(self.ema_model.parameters(), model.parameters(), self.decay)

        # update the buffers with certain exception 
        for (buffer_ema_name, buffer_ema), (buffer_name, buffer) in zip(self.ema_model.named_buffers(), model.named_buffers()):
            if "num_batches_tracked" in buffer_ema_name:
                buffer_ema.copy_(buffer)
            else:
                update_ema([buffer_ema], [buffer], self.decay)

def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, shape, row_index):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    array = np.frombuffer(row_bytes, dtype=dtype)
    
    if len(shape) > 0:
        array = array.reshape(shape)
    return array 

def get_array_shape_from_lmdb(lmdb_env, array_name):
    with lmdb_env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))

    return image_shape 

def create_image_grid(args, images_array, captions=None):
    # Set the dimensions of each individual image
    thumbnail_width = args.image_resolution
    thumbnail_height = args.image_resolution 

    # Spacing and margins
    caption_height = 30
    spacing = 15
    images_per_row = int(len(images_array) ** (1/2))  

    # Calculate grid dimensions
    total_width = (thumbnail_width + spacing) * images_per_row
    total_height = (thumbnail_height + caption_height + spacing) * (len(images_array) // images_per_row)

    # Create the big grid image with white background
    grid_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)

    # Load a font for the captions
    font = ImageFont.load_default()

    # Populate the grid with images and captions
    if captions is None:
        captions = ["" for _ in range(len(images_array))]

    for i, (img_data, caption) in enumerate(zip(images_array, captions)):
        img = Image.fromarray(img_data)
        img.thumbnail((thumbnail_width, thumbnail_height))

        # Calculate position in the grid
        x = (i % images_per_row) * (thumbnail_width + spacing)
        y = (i // images_per_row) * (thumbnail_height + caption_height + spacing)

        # Paste image and draw caption
        grid_img.paste(img, (x, y))

        wrapped_caption = textwrap.fill(str(caption), width=80)

        draw.text((x, y + thumbnail_height), f"{i:05d}_{wrapped_caption}", font=font, fill=(0, 0, 0))

    return grid_img 

class SDTextDataset(Dataset):
    def __init__(self, anno_path, tokenizer_one, is_sdxl=False, tokenizer_two=None):
        if anno_path.endswith(".txt"):
            self.all_prompts = []
            with open(anno_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue 
                    else:
                        self.all_prompts.append(line)
        else:
            self.all_prompts = pickle.load(open(anno_path, "rb"))
    
        self.all_indices = list(range(len(self.all_prompts)))

        self.is_sdxl = is_sdxl # sdxl uses two tokenizers
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        print(f"Loaded {len(self.all_prompts)} prompts")

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx]
        if prompt == None:
            prompt = ""


        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        output_dict = {
            'index': self.all_indices[idx],
            'key': prompt,
            'text_input_ids_one': text_input_ids_one,
        }

        if self.is_sdxl:
            text_input_ids_two = self.tokenizer_two(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            output_dict['text_input_ids_two'] = text_input_ids_two

        return output_dict 
    
def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample

def get_prev_sample_from_noise(sample, model_output, alphas, betas, timestep):
    alpha_t = alphas[timestep]
    beta_t = betas[timestep]
    pred_latents = (sample - beta_t * model_output) / alpha_t
    
    return pred_latents

class NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 1)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def extract_text_embeddings(batch, accelerator, text_encoder_one, text_encoder_two):
    text_input_ids_one = batch['text_input_ids_one'].to(accelerator.device).squeeze(1)
    text_input_ids_two = batch['text_input_ids_two'].to(accelerator.device).squeeze(1)
    prompt_embeds_list = []

    for text_input_ids, text_encoder in zip([text_input_ids_one, text_input_ids_two], [text_encoder_one, text_encoder_two]):
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    # use the second text encoder's pooled prompt embeds  (the first value is overwrited)
    pooled_prompt_embeds = pooled_prompt_embeds.view(len(text_input_ids_one), -1) 

    return prompt_embeds, pooled_prompt_embeds

class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super(EdgeDetectionModel, self).__init__()
        # Sobel filters for edge detection
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        sobel_x_kernel = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]])
        sobel_y_kernel = torch.tensor([[-1., -2., -1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  2.,  1.]])
        
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel.view(1, 1, 3, 3))
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel.view(1, 1, 3, 3))
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            x = transforms.Grayscale()(x)
        
        # Apply Sobel filters
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        
        # Calculate gradient magnitude (edge detection result)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        
        return edges
    
def total_variation_loss(x):
    """
    计算图像 x 的 TV loss
    :param x: 图像 tensor，形状为 (batch_size, channels, height, width)
    :return: TV Loss
    """
    # 计算水平方向的差异
    diff_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    # 计算垂直方向的差异
    diff_y = x[:, :, :, 1:] - x[:, :, :, :-1]

    # 计算 L1 范数
    loss = torch.abs(diff_x)[..., :-1] + torch.abs(diff_y)[..., :-1, :]

    return loss

# def attention_diversification_loss(attn, th=2):
#     sim_sum = 0
#     counter = 1e-6
#     for i in range(len(attn)):
#         mask0 = attn[i].mean(dim=1).squeeze()
#         n_tokens = mask0.shape[-1]
#         threshold = th/n_tokens
#         score0 = torch.mean(mask0, dim=1, keepdim=True)
#         mask0 = (mask0 > threshold) * (mask0)
#         score0 = (score0 > threshold) * (score0)
#         sim = torch.nn.functional.cosine_similarity(score0, mask0, dim=-1)
#         sim = sim.mean()
#         sim_sum += sim
#         counter += 1
#     sim_sum = sim_sum / counter
#     return sim_sum

def attention_diversification_loss(feature, th=2):
    B, L, N, D = feature.shape

    avg_feature = torch.mean(feature, dim=2, keepdim=True).repeat(1, 1, N, 1)

    sim = torch.nn.functional.cosine_similarity(feature, avg_feature, dim=-1)

    sim = (sim ** 2).mean(dim=-1)

    # sim = sim.mean(dim=-1)

    return sim.mean()


def token_similarity_loss(attn):
    """
    :param attn: (batch, layers, tokens, D)
    """
    B, L, N, D = attn.shape

    attn = torch.nn.functional.normalize(attn, p=2, dim=-1)

    cosine_similarities = torch.bmm(attn.view(B * L, N, D), attn.view(B * L, D, N).transpose(1, 2))

    cosine_similarity_matrix = cosine_similarities.view(B, L, N, N)

    cosine_similarity_matrix.fill_diagonal_(0) # (B, L, N, N)

    loss = torch.mean(cosine_similarity_matrix)  
    
    return loss


def spectral_penalty(latent, periods_lat=(2,), bandwidth=0.15, weight=1.0, anisotropic=False, savepath=None, idx=None):
    """
    latent: [B, C, H, W] —— 冻结 decoder 之前的最终 latent
    periods_lat: 需要抑制的“潜空间周期”列表（像素），例如 (2,) 或 (2,4)
    bandwidth: 频带相对宽度（0.1~0.25），越大越宽
    anisotropic: True 时更偏向抑制水平/垂直条纹（十字形）；False 为环形带阻
    返回：标量 loss
    """
    latent = latent.to(torch.float32)  # Convert to float32 for FFT
    B, C, H, W = latent.shape
    # 去均值防止 DC 分量干扰
    x = latent - latent.mean(dim=(-2, -1), keepdim=True)

    # 2D FFT -> 功率谱
    X = torch.fft.fft2(x, dim=(-2, -1))
    P = (X.real**2 + X.imag**2) / (H * W)  # [B, C, H, W]

    # 频率网格（以 0 为中心）
    fy = torch.fft.fftfreq(H, d=1.0).to(latent.device)  # [-0.5,0.5)
    fx = torch.fft.fftfreq(W, d=1.0).to(latent.device)
    fy, fx = torch.meshgrid(fy, fx, indexing='ij')      # [H, W]
    rho = torch.sqrt(fx**2 + fy**2)                     # 半径

    mask = torch.zeros((H, W), device=latent.device)
    for p in periods_lat:
        # 周期 p -> 归一化频率 f=1/p，对应半径 r=1/p
        r0 = 1.0 / float(p)
        bw = bandwidth * r0

        if anisotropic:
            # 水平/垂直“十字”两个带阻：|fx|≈r0 或 |fy|≈r0
            hx = torch.exp(-0.5 * ((torch.abs(fx) - r0) / (bw + 1e-8))**2)
            hy = torch.exp(-0.5 * ((torch.abs(fy) - r0) / (bw + 1e-8))**2)
            mask = torch.maximum(mask, torch.maximum(hx, hy))
        else:
            # 环形带阻：|rho|≈r0
            ring = torch.exp(-0.5 * ((rho - r0) / (bw + 1e-8))**2)
            mask = torch.maximum(mask, ring)

    # 归一化，避免随图尺寸变化
    mask = mask / (mask.max().clamp_min(1e-6))

    if savepath is not None:
        mod100 = idx // 100 * 100
        os.makedirs(f"{savepath}/{mod100:04d}/{idx:04d}", exist_ok=True)
        savepath = f"{savepath}/{mod100:04d}/{idx:04d}"
        plot_channel_log_histograms(P, savepath, f"histP")
        plot_channel_box_plots(P, f"{savepath}/box.png")

    # 在带阻区域内惩罚功率
    loss_spec = (P * mask).sum() * weight
    return loss_spec


def plot_channel_histograms(tensor, out_dir, savenames):
    """
    Plots a separate histogram for each channel of a given tensor.

    This function is designed to handle tensors of shape (H, W), (C, H, W),
    or (B, C, H, W). It flattens the H and W dimensions for each channel
    and creates a histogram to visualize the distribution of values.

    Args:
        tensor (torch.Tensor): The input tensor.
        out_dir (str): The directory to save the histogram images.
        savenames (str): The base name to save the image files.
    """
    os.makedirs(out_dir, exist_ok=True)
    num_dims = tensor.ndimension()
    tensor_np = tensor.detach().cpu().numpy()

    def _plot_single_histogram(data, save_path, title):
        """Helper function to create and save a single histogram plot."""
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=50, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Histogram saved to {save_path}")

    if num_dims == 4:  # Shape (B, C, H, W)
        B, C, H, W = tensor_np.shape
        # Iterate over batch and channels
        for b in range(B):
            for c in range(C):
                flattened_data = tensor_np[b, c].flatten()
                save_path = os.path.join(out_dir, f"{savenames}_b{b}_c{c}.png")
                _plot_single_histogram(flattened_data, save_path, f'Batch {b}, Channel {c} Histogram')

    elif num_dims == 3:  # Shape (C, H, W)
        C, H, W = tensor_np.shape
        # Iterate over channels
        for c in range(C):
            flattened_data = tensor_np[c].flatten()
            save_path = os.path.join(out_dir, f"{savenames}_c{c}.png")
            _plot_single_histogram(flattened_data, save_path, f'Channel {c} Histogram')

    elif num_dims == 2:  # Shape (H, W) - Treat as a single channel
        flattened_data = tensor_np.flatten()
        save_path = os.path.join(out_dir, f"{savenames}_c0.png")
        _plot_single_histogram(flattened_data, save_path, 'Single Channel Histogram')

    else:
        print(f"Warning: Input tensor with {num_dims} dimensions is not supported for channel-wise plotting.")
        print("Falling back to plotting a single histogram for the whole tensor.")
        flattened_data = tensor_np.flatten()
        save_path = os.path.join(out_dir, f"{savenames}_all_flattened.png")
        _plot_single_histogram(flattened_data, save_path, 'All Data Flattened Histogram')

def analyze_channel_distribution(tensor, out_dir, savenames, zero_threshold=1e-6):
    """
    Analyzes the distribution of each channel of a tensor and provides a text summary.

    This function is designed to handle tensors of shape (H, W), (C, H, W),
    or (B, C, H, W). It flattens the H and W dimensions for each channel
    and provides a detailed text report of its statistics.

    Args:
        tensor (torch.Tensor): The input tensor to analyze.
        out_dir (str): The directory to save the text report files.
        savenames (str): The base name for the saved files.
        zero_threshold (float): Values below this threshold are considered zero.
    """
    # Create the directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Detach from computation graph and move to CPU
    tensor_np = tensor.detach().cpu().numpy()
    
    # Determine the number of dimensions
    num_dims = tensor.ndimension()
    
    # Helper function to generate and save a single channel report
    def _generate_report(data, filename, title):
        non_zero_indices = np.abs(data) > zero_threshold
        num_total = data.size
        num_non_zero = np.sum(non_zero_indices)
        
        report_lines = [
            f"--- {title} Analysis ---",
            f"Total elements: {num_total:,}",
            f"Global Mean: {np.mean(data):.4f}",
            f"Global Standard Deviation: {np.std(data):.4f}",
            "---"
        ]

        if num_non_zero == 0:
            report_lines.append("All elements are zero.")
        else:
            non_zero_data = data[non_zero_indices]
            report_lines.append(f"Number of non-zero elements: {num_non_zero:,} ({num_non_zero / num_total:.2%})")
            report_lines.append(f"Non-zero Min Value: {np.min(non_zero_data):.4f}")
            report_lines.append(f"Non-zero Max Value: {np.max(non_zero_data):.4f}")
            report_lines.append(f"Non-zero Mean: {np.mean(non_zero_data):.4f}")
            report_lines.append(f"Non-zero Median: {np.median(non_zero_data):.4f}")
            report_lines.append(f"Non-zero Std Dev: {np.std(non_zero_data):.4f}")

        report = "\n".join(report_lines)
        save_path = os.path.join(out_dir, filename)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Analysis report saved to: {save_path}")

    # Process based on tensor dimensions
    if num_dims == 4:  # Shape (B, C, H, W)
        B, C, _, _ = tensor_np.shape
        for b in range(B):
            for c in range(C):
                flattened_data = tensor_np[b, c].flatten()
                filename = f"{savenames}_b{b}_c{c}_report.txt"
                _generate_report(flattened_data, filename, f"Batch {b}, Channel {c} Tensor")

    elif num_dims == 3:  # Shape (C, H, W)
        C, _, _ = tensor_np.shape
        for c in range(C):
            flattened_data = tensor_np[c].flatten()
            filename = f"{savenames}_c{c}_report.txt"
            _generate_report(flattened_data, filename, f"Channel {c} Tensor")

    elif num_dims == 2:  # Shape (H, W)
        flattened_data = tensor_np.flatten()
        filename = f"{savenames}_single_channel_report.txt"
        _generate_report(flattened_data, filename, "Single Channel Tensor")
        
    else:
        print(f"Warning: Input tensor with {num_dims} dimensions is not supported for channel-wise analysis.")
        print("Please provide a tensor with 2, 3, or 4 dimensions.")

def plot_channel_log_histograms(tensor, out_dir, savenames):
    """
    Plots a separate logarithmic histogram for each channel of a given tensor.

    Args:
        tensor (torch.Tensor): The input tensor.
        out_dir (str): The directory to save the histogram images.
        savenames (str): The base name to save the image files.
    """
    os.makedirs(out_dir, exist_ok=True)
    tensor_np = tensor.detach().cpu().numpy()

    num_dims = tensor.ndimension()
    
    def _plot_log_histogram(data, save_path, title):
        plt.figure(figsize=(10, 6))
        
        # Use a logarithmic scale on the y-axis
        plt.hist(data, bins=50, color='skyblue', edgecolor='black', log=True)
        
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency (Log Scale)')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Logarithmic histogram saved to {save_path}")

    if num_dims == 4:
        B, C, _, _ = tensor_np.shape
        for b in range(B):
            for c in range(C):
                data = tensor_np[b, c].flatten()
                save_path = os.path.join(out_dir, f"{savenames}_b{b}_c{c}_loghist.png")
                _plot_log_histogram(data, save_path, f'Log Histogram of Batch {b}, Channel {c}')

    elif num_dims == 3:
        C, _, _ = tensor_np.shape
        for c in range(C):
            data = tensor_np[c].flatten()
            save_path = os.path.join(out_dir, f"{savenames}_c{c}_loghist.png")
            _plot_log_histogram(data, save_path, f'Log Histogram of Channel {c}')

    elif num_dims == 2:
        data = tensor_np.flatten()
        save_path = os.path.join(out_dir, f"{savenames}_loghist.png")
        _plot_log_histogram(data, save_path, 'Log Histogram of Single Channel')
    
    else:
        print(f"Warning: Tensor with {num_dims} dimensions is not supported for channel-wise plotting.")

def plot_channel_box_plots(tensor, save_path, title="Tensor Channel Value Distribution"):
    """
    Plots a box plot for each channel of a given tensor on a single graph.

    This function is designed to handle tensors of shape (H, W), (C, H, W),
    or (B, C, H, W). It aggregates data from each channel and displays
    a comparative box plot, ideal for visualizing and comparing
    distribution characteristics like median, quartiles, and outliers.

    Args:
        tensor (torch.Tensor): The input tensor.
        save_path (str): The path to save the box plot image.
        title (str, optional): The title of the box plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Detach from computation graph and move to CPU
    tensor_np = tensor.detach().cpu().numpy()
    
    num_dims = tensor.ndimension()
    
    # Prepare data for plotting
    data_to_plot = []
    labels = []
    
    if num_dims == 4:
        B, C, _, _ = tensor_np.shape
        for b in range(B):
            for c in range(C):
                data_to_plot.append(tensor_np[b, c].flatten())
                labels.append(f"B{b} C{c}")
                
    elif num_dims == 3:
        C, _, _ = tensor_np.shape
        for c in range(C):
            data_to_plot.append(tensor_np[c].flatten())
            labels.append(f"C{c}")
            
    elif num_dims == 2:
        data_to_plot.append(tensor_np.flatten())
        labels.append("C0")
    
    else:
        print(f"Warning: Input tensor with {num_dims} dimensions is not supported for channel-wise plotting.")
        return

    # Create the box plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(data_to_plot, labels=labels, showfliers=True)
    
    # Add titles and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Channel', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, axis='y', alpha=0.75)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Box plot saved to {save_path}")
