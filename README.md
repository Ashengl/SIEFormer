# SIEFormer
Spectral-Interpretable and -Enhanced Transformer for Generalized Category Discovery

## Installation
```shell
$ cd repository
$ pip install -r requirements.txt
```

## Datasets
The datasets we use are:

[CUB-200](https://www.vision.caltech.edu/datasets/cub_200_2011/), [StanfordCars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset), [Aircraft](https://www.kaggle.com/code/metamath/fgvc-aircraft),  [Herbarium](https://www.kaggle.com/competitions/herbarium-2019-fgvc6/data),[Cifar](), [ImageNet-100](https://www.kaggle.com/competitions/herbarium-2019-fgvc6/data).

The split of datasets follows [GCD](https://github.com/sgvaze/generalized-category-discovery).



<details>
  <summary>The index of Imagenet-100 from Image-1k contains：</summary>
  <pre><code>[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 16, 18, 19, 20, 21, 22, 24, 26, 28, 29, 32, 33, 34, 35, 36, 38, 39, 41, 42, 46, 48, 50, 52, 54, 55, 56, 57, 59, 60, 61, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 83, 84, 88, 89, 90, 91, 92, 93, 94, 96, 97, 99, 100, 104, 106, 107, 108, 110, 111, 112, 113, 115, 116, 117, 118, 119, 123, 124, 125, 127, 129, 130, 133, 134, 135, 137, 138, 140, 141, 143, 144, 146, 150]</p></code></pre>
</details>

## How to start SIEFormer with [GCD](https://github.com/sgvaze/generalized-category-discovery)
* Replace **vision_transformer.py** with the provided model or change the calculation of Attention to:
<details>
  <summary>code</summary>
  <pre><code>
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_v1 = nn.Linear(dim, dim, bias=True)
        self.proj_v2 = nn.Linear(dim, dim, bias=True)
        self.complex_weight = nn.Parameter(torch.cat((torch.ones(1, 1, 1, head_dim//2 + 1, 1), torch.zeros(1, 1, 1, head_dim//2 + 1, 1)), dim=4))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ac = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        B, N, C = x.shape  # 48, 768, 197
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, 48, 12, 197, 64
        q, k, v = qkv[0], qkv[1], qkv[2]  # 48, 12, 197, 64
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 48, 12, 197, 197
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 48, 12, 197, 197
        attn_hat = (q.transpose(1, 2).reshape(B, N, C) @ k.transpose(2, 3).reshape(B, C, N))
        attn_hat = (attn_hat + attn_hat.transpose(1, 2))/2
        attn_hat = self.ac(attn_hat)
        attn_hat_d = torch.sum(attn_hat, dim=2)
        attn_hat_d[attn_hat_d != 0] = torch.sqrt(1.0 / attn_hat_d[attn_hat_d != 0])
        Norm_attn_hat = attn_hat * attn_hat_d.unsqueeze(1) * attn_hat_d.unsqueeze(2)
        I = torch.eye(Norm_attn_hat.size(1)).cuda().unsqueeze(0)
        L = I - Norm_attn_hat  # 0,2
        L_2 = torch.bmm(L - I, L - I)
        out = self.proj_v1(torch.bmm(L_2 - I, v.contiguous().transpose(1, 2).reshape(B, N, C)))
        out = out - self.proj_v2(torch.bmm(L_2, v.contiguous().transpose(1, 2).reshape(B, N, C)))
        fft_v = torch.fft.rfft(v.contiguous(), dim=3, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        fft_v = fft_v * weight
        ifft_v = torch.fft.irfft(fft_v, dim=3, norm='ortho')
        x = (attn @ ifft_v).transpose(1, 2).reshape(B, N, C)  # 48, 768, 197
        x = self.proj(x)
        x = self.proj_drop(x)
        return x + out, attn
  </code></pre>
</details>

* Replace **contrastive_training.py** with the provided training code or set parameters **require_grad = True**:
<details>
  <summary>code</summary>
  <pre><code>
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
                if 'proj_v' in name:
                    m.requires_grad = True
                    torch.nn.init.zeros_(m)
                if 'complex' in name:
                    m.requires_grad = True
  </code></pre>
</details>

* Start your SIEFormer

## Train
If you wish to try training your SIEFormer, please run contrastive_training, for example:
<details>
  <summary>script</summary>
  <pre><code>
python -m methods.contrastive_training.contrastive_training \
            --dataset_name 'cifar100' \
            --batch_size 40 \
            --grad_from_block 11 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 8 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-7 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.0005 \
            --eval_funcs 'v1' 'v2' \
  </code></pre>
</details>

Or, just change your script as:
<details>
  <summary>code</summary>
  <pre><code>
    batch_size = 40
    learning_rate = 0.0005
    sup_con_weight = 0.35 for generic dataset, 
                     1.0 for fine-grained dataset
    weight_decay = 5e-7
  </code></pre>
</details>

## Acknowledgement

Our codes are based on [Generalized Category Discovery](https://github.com/sgvaze/generalized-category-discovery).
