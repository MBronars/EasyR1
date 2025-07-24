import re
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
import json


def extract_traj(response):
    # regex to capture the two numbers inside each [x, y]
    points = re.findall(r'\[ *(\d+)\s*,\s*(\d+) *\]', response)

    # convert to list of int-tuples
    points = [(int(x), int(y)) for x, y in points]
    return points


def exp_norm(value, k=5):
    return torch.exp(-k * value)


def format_reward_helper(text: str):
    # 1. Check tags
    tag_match = re.fullmatch(r"\s*<answer>([\s\S]*)</answer>\s*", text)
    tags_ok = bool(tag_match)
    if not tags_ok:
        return False, False, False

    inner = tag_match.group(1).strip()
    
    # 2. Check first word
    parts = inner.split(None, 1)
    gripper_ok = parts and parts[0].lower() in ("open", "close")
    if not gripper_ok:
        return tags_ok, False, False

    # 3. Extract and validate JSON block
    m = re.search(r"```json\s*\n([\s\S]*?)\s*```", inner)
    if not m:
        return tags_ok, gripper_ok, False

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return tags_ok, gripper_ok, False

    # Must be a list
    if not isinstance(data, list):
        return tags_ok, gripper_ok, False

    # Validate entries and ordering
    for idx, entry in enumerate(data):
        if (
            not isinstance(entry, dict)
            or set(entry.keys()) != {"point_2d", "label"}
            or not isinstance(entry["point_2d"], list)
            or len(entry["point_2d"]) != 2
            or not all(isinstance(n, int) for n in entry["point_2d"])
            or entry["label"] != f"trajectory_{idx}"
        ):
            return tags_ok, gripper_ok, False

    return tags_ok, gripper_ok, True

def format_reward(text: str):
    """
    Check if the response is in the correct format.
    Returns a score of 1.0 if the format is correct, otherwise 0.0.
    """
    tags_ok, gripper_ok, json_ok = format_reward_helper(text)
    
    if not tags_ok or not gripper_ok or not json_ok:
        return 0.0

    return 1.0


def endpoint_reward(pred_traj: torch.Tensor, gt_traj: torch.Tensor, k=10):
    """
    Distance between predicted and GT trajectory endpoints.
        pred_traj: (N, 3)
        gt_traj: (M, 3)
    """
    p_hat_N = pred_traj[-1]
    p_star_M = gt_traj[-1]

    diff = p_hat_N - p_star_M
    dist_squared = torch.sum(diff ** 2, dim=-1)
    reward = exp_norm(dist_squared, k=k)
    return reward


def DFD_reward(pred_traj: torch.Tensor, gt_traj: torch.Tensor, k=5):
    """
    Compute Discrete Fréchet Distance(DFD) between predicted and GT trajectories.
        pred_traj: (N, 3)
        gt_traj: (M, 3)
    """
    N, M = pred_traj.shape[0], gt_traj.shape[0]
    D = torch.cdist(pred_traj.unsqueeze(0), gt_traj.unsqueeze(0), p=2)[0]  # (N, M)

    dp = torch.zeros((N, M), device=pred_traj.device)
    dp[0, 0] = D[0, 0]

    for i in range(1, N):
        dp[i, 0] = torch.max(dp[i - 1, 0], D[i, 0])
    for j in range(1, M):
        dp[0, j] = torch.max(dp[0, j - 1], D[0, j])

    for i in range(1, N):
        for j in range(1, M):
            dp[i, j] = torch.max(
                torch.min(torch.stack(
                    [dp[i - 1, j],     # ←
                    dp[i - 1, j - 1],  # ↖
                    dp[i, j - 1],]     # ↑
                )),
                D[i, j]
            )

    DFD = dp[N - 1, M - 1]
    return exp_norm(DFD)


def HD_reward(pred_traj: torch.Tensor, gt_traj: torch.Tensor, k=5):
    """
    Compute the Hausdorff Distance(HD) between predicted and GT trajectories.
        pred_traj: (N, 3)
        gt_traj: (M, 3)
    """
    dists_1_to_2 = torch.cdist(pred_traj, gt_traj, p=2)  # [T1, T2]
    d1 = dists_1_to_2.min(dim=1).values.max()  # max over min distances from traj1 to traj2
    d2 = dists_1_to_2.min(dim=0).values.max()  # max over min distances from traj2 to traj1
    HD = torch.max(d1, d2)
    return exp_norm(HD, k=k)


def RMSE_reward(pred_traj: torch.Tensor, gt_traj: torch.Tensor, k=5):
    """
    Compute the RMSE between predicted and GT trajectories.
        pred_traj: (N, 3)
        gt_traj: (M, 3)
    """
    # N = len(pred_traj)
    M = len(gt_traj)

    # interpolate pred trajectory to match the length of gt
    # aligning the two trajs while retaining gt information
    pred_traj_interp: torch.Tensor = F.interpolate(pred_traj.T.unsqueeze(0), size=M, mode='linear', align_corners=True)
    pred_traj_interp = pred_traj_interp.squeeze(0).T
    RMSE = torch.sqrt(torch.mean(torch.sum((pred_traj_interp - gt_traj) ** 2, dim=-1)))
    return exp_norm(RMSE, k=k)


def trajectory_reward(pred_traj: torch.Tensor, gt_traj: torch.Tensor):
    return DFD_reward(pred_traj, gt_traj) + HD_reward(pred_traj, gt_traj) + RMSE_reward(pred_traj, gt_traj)


def compute_score(
        reward_inputs: List[Dict[str, Any]], 
        format_weight: float = 1.0,
        ) -> List[Dict[str, float]]:
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        format_score = format_reward(response)

        pred_traj = extract_traj(response)
        gt_traj = extract_traj(reward_input["ground_truth"])
        trajectory_score = trajectory_reward(pred_traj, gt_traj)

        scores.append(
            {
                "overall": trajectory_score + format_weight * format_score,
                "format": format_score,
                "trajectory": trajectory_score
            }
        )

    return scores