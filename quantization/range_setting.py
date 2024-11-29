import torch
import torch.nn.functional as F

def max_min_range_settings(input_float):
    if isinstance(input_float, (list, tuple)):
        return min(input_float), max(input_float)
    elif isinstance(input_float, torch.Tensor):
        return input_float.min(), input_float.max()

def MSE_range_setting(input_float, bit_width, num_candidates = 100):
    if isinstance(input_float, (list, tuple)):
        x_min, x_max = min(input_float), max(input_float)
    elif isinstance(input_float, torch.Tensor):
        x_min, x_max = input_float.min(), input_float.max()

    candidates = torch.linspace(x_min, x_max, num_candidates)
    MSE_opt = float('inf')
    q_min_opt, q_max_opt = x_min, x_max

    for q_min in candidates:
        for q_max in candidates:
            if q_max <= q_min:
                continue

            scale_factor = abs(q_max - q_min) / (2 ** bit_width - 1)
            zero_point = -q_min / scale_factor

            V_quan = torch.clamp(torch.round(input_float / scale_factor) + zero_point, 0, 2 ** bit_width - 1)
            V_hat = scale_factor * (V_quan - zero_point)

            MSE_now = (torch.norm(input=input_float - V_hat, p='fro')) ** 2

            if MSE_now < MSE_opt:
                MSE_opt = MSE_now
                q_min_opt, q_max_opt = q_min, q_max

    return q_min_opt, q_max_opt

def CE_range_setting(input_image, bit_width, model, num_candidates = 100):
    logits_vec = model(input_image)

    # initial range
    if isinstance(logits_vec, list) or isinstance(logits_vec, tuple):
        x_min, x_max = min(logits_vec), max(logits_vec)
    elif isinstance(logits_vec, torch.Tensor):
        x_min, x_max = logits_vec.min(), logits_vec.max()

    # generate all possible ranges
    candidates = torch.linspace(x_min, x_max, num_candidates)

    CE_opt = float('inf')
    q_min_opt, q_max_opt = x_min, x_max

    for q_min in candidates:
        for q_max in candidates:
            if q_max <= q_min:
                continue
            scale_factor = abs(q_max - q_min) / (2 ** bit_width - 1)
            zero_point = -q_min / scale_factor

            logits_vec_quan = torch.clamp(torch.round(logits_vec / scale_factor) + zero_point, 0, 2 ** bit_width - 1)
            logits_vec_hat = scale_factor * (logits_vec_quan - zero_point)

            # since S(softmax(logits_vec)) = const -> use KL-D to replace CE
            logits_vec_softmax = F.softmax(logits_vec, dim=1)
            logits_vec_hat_softmax = F.softmax(logits_vec_hat, dim = 1)
            KL_D = F.kl_div(logits_vec_softmax, logits_vec_hat_softmax, reduce="batchmean")
            # CE = F.cross_entropy(logits_vec_softmax, logits_vec_hat_softmax)
            # CE = F.kl_div(logits_vec_softmax, logits_vec_hat_softmax, reduce="batchmean")

            if KL_D < CE_opt:
                CE_opt = KL_D
                q_min_opt, q_max_opt = q_min, q_max

    return q_min_opt, q_max_opt