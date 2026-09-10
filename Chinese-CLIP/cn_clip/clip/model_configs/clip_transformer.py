import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerConfig:
    """Transformer模型配置类"""

    def __init__(
            self,
            vocab_size: int = 21128,
            text_attention_probs_dropout_prob: float = 0.1,
            text_hidden_act: str = "gelu",
            text_hidden_dropout_prob: float = 0.1,
            text_hidden_size: int = 768,
            text_initializer_range: float = 0.02,
            text_intermediate_size: int = 3072,
            text_max_position_embeddings: int = 512,
            text_num_attention_heads: int = 12,
            text_num_hidden_layers: int = 12,
            text_type_vocab_size: int = 2,
    ):
        self.vocab_size = vocab_size
        self.attention_probs_dropout_prob = text_attention_probs_dropout_prob
        self.hidden_act = text_hidden_act
        self.hidden_dropout_prob = text_hidden_dropout_prob
        self.hidden_size = text_hidden_size
        self.initializer_range = text_initializer_range
        self.intermediate_size = text_intermediate_size
        self.max_position_embeddings = text_max_position_embeddings
        self.num_attention_heads = text_num_attention_heads
        self.num_hidden_layers = text_num_hidden_layers
        self.type_vocab_size = text_type_vocab_size


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 注意力投影矩阵
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # 缩放因子
        self.scale = self.attention_head_size ** -0.5

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # 线性投影
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)

        # 调整张量形状以适应多头注意力
        query_states = self.transpose_for_scores(query_states)
        key_states = self.transpose_for_scores(key_states)
        value_states = self.transpose_for_scores(value_states)

        # 处理KV缓存（用于生成时加速）
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present = (key_states, value_states) if use_cache else None

        # 计算注意力得分
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            # 应用注意力掩码
            attn_weights = attn_weights + attention_mask

        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), self.all_head_size)

        # 输出投影
        output = self.resid_dropout(self.out_proj(attn_output))
        return output, present


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 选择激活函数
        if config.hidden_act == "gelu":
            self.activation = F.gelu
        elif config.hidden_act == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_h_to_4h(x)
        x = self.activation(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attention = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = MLP(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # 自注意力层
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # 前馈网络层
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present


class TransformerTextEncoder(nn.Module):
    """完整的Transformer文本编码器"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # 嵌入层
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 层归一化和dropout
        self.ln_embed = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 编码器层
        self.encoder = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: bool = False,
    ) -> torch.Tensor:
        # 获取输入形状
        batch_size, seq_length = input_ids.shape

        # 如果没有提供token_type_ids，创建全0的类型ID
        if token_type_ids is None:
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=input_ids.device)

        # 如果没有提供position_ids，创建从0到seq_length-1的位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # 如果没有提供注意力掩码，创建一个全1的掩码（表示所有位置都参与注意力计算）
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=input_ids.device)

        # 将注意力掩码转换为适合注意力计算的形式
        # (batch_size, seq_length) -> (batch_size, 1, 1, seq_length)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # 将0转换为负无穷大（在softmax前应用），1转换为0
        attention_mask = (1.0 - attention_mask) * -10000.0

        # 获取嵌入表示
        inputs_embeds = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 叠加嵌入
        hidden_states = inputs_embeds + position_embeddings + token_type_embeddings
        hidden_states = self.ln_embed(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # 依次通过所有编码器层
        presents = [] if use_cache else None
        for i, layer in enumerate(self.encoder):
            past_key_value = presents[i] if presents is not None else None
            hidden_states, present = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        return hidden_states


# 创建基于给定配置的Transformer模型实例
def create_transformer_model() -> TransformerTextEncoder:
    """创建基于给定配置的Transformer文本编码器实例"""
    config = TransformerConfig(
        vocab_size=21128,
        text_attention_probs_dropout_prob=0.1,
        text_hidden_act="gelu",
        text_hidden_dropout_prob=0.1,
        text_hidden_size=768,
        text_initializer_range=0.02,
        text_intermediate_size=3072,
        text_max_position_embeddings=512,
        text_num_attention_heads=12,
        text_num_hidden_layers=12,
        text_type_vocab_size=2,
    )
    return TransformerTextEncoder(config)


# 示例：如何使用该模型
if __name__ == "__main__":
    # 创建模型
    model = create_transformer_model()

    # 准备输入
    input_ids = torch.randint(0, 21128, (1, 32))  # 批次大小为1，序列长度为32
    attention_mask = torch.ones((1, 32))

    # 前向传播
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output.shape}")  # 应该是 (1, 32, 768)