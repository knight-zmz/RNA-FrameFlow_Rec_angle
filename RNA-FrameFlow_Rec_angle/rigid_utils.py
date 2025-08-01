# -------------------------------------------------------------------------------------------------------------------------------------
# 以下代码改编自 se3_diffusion (https://github.com/jasonkyuyim/se3_diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------
"""OpenFold的向量更新函数的变体，支持掩码。"""

import numpy as np
import torch
from beartype.typing import Any, Callable, List, Optional, Tuple, Union
from jaxtyping import Float

NODE_MASK_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes"]
UPDATE_NODE_MASK_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 1"]
QUATERNION_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 4"]
ROTATION_TENSOR_TYPE = Float[torch.Tensor, "... 3 3"]
COORDINATES_TENSOR_TYPE = Float[torch.Tensor, "... num_nodes 3"]


def rot_matmul(a: ROTATION_TENSOR_TYPE, b: ROTATION_TENSOR_TYPE) -> ROTATION_TENSOR_TYPE:
    """对两个旋转矩阵张量进行矩阵乘法。手写实现以避免AMP降精度。

    参数:
        a: [*, 3, 3] 左乘数
        b: [*, 3, 3] 右乘数
    返回:
        ab的乘积
    """
    row_1 = torch.stack(
        [
            a[..., 0, 0] * b[..., 0, 0]
            + a[..., 0, 1] * b[..., 1, 0]
            + a[..., 0, 2] * b[..., 2, 0],
            a[..., 0, 0] * b[..., 0, 1]
            + a[..., 0, 1] * b[..., 1, 1]
            + a[..., 0, 2] * b[..., 2, 1],
            a[..., 0, 0] * b[..., 0, 2]
            + a[..., 0, 1] * b[..., 1, 2]
            + a[..., 0, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_2 = torch.stack(
        [
            a[..., 1, 0] * b[..., 0, 0]
            + a[..., 1, 1] * b[..., 1, 0]
            + a[..., 1, 2] * b[..., 2, 0],
            a[..., 1, 0] * b[..., 0, 1]
            + a[..., 1, 1] * b[..., 1, 1]
            + a[..., 1, 2] * b[..., 2, 1],
            a[..., 1, 0] * b[..., 0, 2]
            + a[..., 1, 1] * b[..., 1, 2]
            + a[..., 1, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_3 = torch.stack(
        [
            a[..., 2, 0] * b[..., 0, 0]
            + a[..., 2, 1] * b[..., 1, 0]
            + a[..., 2, 2] * b[..., 2, 0],
            a[..., 2, 0] * b[..., 0, 1]
            + a[..., 2, 1] * b[..., 1, 1]
            + a[..., 2, 2] * b[..., 2, 1],
            a[..., 2, 0] * b[..., 0, 2]
            + a[..., 2, 1] * b[..., 1, 2]
            + a[..., 2, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )

    return torch.stack([row_1, row_2, row_3], dim=-2)


def rot_vec_mul(r: ROTATION_TENSOR_TYPE, t: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
    """对向量应用旋转。手写实现以避免AMP降精度。

    参数:
        r: [*, 3, 3] 旋转矩阵
        t: [*, 3] 坐标张量
    返回:
        [*, 3] 旋转后的坐标
    """
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


def identity_rot_mats(
    batch_dims: Union[Union[Tuple[int], Tuple[np.int64]], torch.Size],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> ROTATION_TENSOR_TYPE:
    """生成单位旋转矩阵。"""
    rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)

    return rots


def identity_trans(
    batch_dims: Union[Union[Tuple[int], Tuple[np.int64]], torch.Size],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> COORDINATES_TENSOR_TYPE:
    """生成零平移向量。"""
    trans = torch.zeros((*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return trans


def identity_quats(
    batch_dims: Union[Union[Tuple[int], Tuple[np.int64]], torch.Size],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    requires_grad: bool = True,
) -> QUATERNION_TENSOR_TYPE:
    """生成单位四元数。"""
    quat = torch.zeros((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)

    with torch.no_grad():
        quat[..., 0] = 1

    return quat


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs: List[Tuple[str, int]]) -> np.ndarray:
    mat = np.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_QTR_MAT = np.zeros((4, 4, 3, 3))
_QTR_MAT[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_QTR_MAT[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_QTR_MAT[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_QTR_MAT[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_QTR_MAT[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_QTR_MAT[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_QTR_MAT[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_QTR_MAT[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_QTR_MAT[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: QUATERNION_TENSOR_TYPE) -> ROTATION_TENSOR_TYPE:
    """将四元数转换为旋转矩阵。

    参数:
        quat: [*, 4] 四元数
    返回:
        [*, 3, 3] 旋转矩阵
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [4, 4, 3, 3]
    mat = quat.new_tensor(_QTR_MAT, requires_grad=False)

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = mat.view((1,) * len(quat.shape[:-2]) + mat.shape)
    quat = quat[..., None, None] * shaped_qtr_mat

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(rot: ROTATION_TENSOR_TYPE) -> QUATERNION_TENSOR_TYPE:
    """将旋转矩阵转换为四元数。"""
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    k = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    _, vectors = torch.linalg.eigh(k)
    return vectors[..., -1]


_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]


def quat_multiply(
    quat1: QUATERNION_TENSOR_TYPE, quat2: QUATERNION_TENSOR_TYPE
) -> QUATERNION_TENSOR_TYPE:
    """四元数相乘。"""
    mat = quat1.new_tensor(_QUAT_MULTIPLY)
    reshaped_mat = mat.view((1,) * len(quat1.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat * quat1[..., :, None, None] * quat2[..., None, :, None], dim=(-3, -2)
    )


def quat_multiply_by_vec(
    quat: QUATERNION_TENSOR_TYPE, vec: COORDINATES_TENSOR_TYPE
) -> QUATERNION_TENSOR_TYPE:
    """四元数与纯向量四元数相乘。"""
    mat = quat.new_tensor(_QUAT_MULTIPLY_BY_VEC)
    reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
    return torch.sum(
        reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None], dim=(-3, -2)
    )


def invert_rot_mat(rot_mat: ROTATION_TENSOR_TYPE) -> ROTATION_TENSOR_TYPE:
    """旋转矩阵求逆。"""
    return rot_mat.transpose(-1, -2)


def invert_quat(
    quat: QUATERNION_TENSOR_TYPE, mask: Optional[NODE_MASK_TENSOR_TYPE] = None
) -> QUATERNION_TENSOR_TYPE:
    """四元数求逆。"""
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    if mask is not None:
        # avoid creating NaNs with masked nodes' "missing" values via division by zero
        inv, quat_mask = quat_prime, mask.bool()
        inv[quat_mask] = inv[quat_mask] / torch.sum(quat[quat_mask] ** 2, dim=-1, keepdim=True)
    else:
        inv = quat_prime / torch.sum(quat**2, dim=-1, keepdim=True)
    return inv


class Rotation:
    """三维旋转。

    根据初始化方式，旋转可以用旋转矩阵或四元数表示，辅助函数可互转。
    为简化梯度计算，底层格式不可原地更改。设计为类似torch Tensor的行为。
    """

    def __init__(
        self,
        rot_mats: Optional[ROTATION_TENSOR_TYPE] = None,
        quats: Optional[QUATERNION_TENSOR_TYPE] = None,
        quats_mask: Optional[NODE_MASK_TENSOR_TYPE] = None,
        normalize_quats: bool = True,
    ):
        """
        初始化Rotation对象，表示3D旋转

        参数:
            rot_mats: 旋转矩阵张量，形状为[*, 3, 3]。与quats参数互斥，只能指定其中一个
            quats: 四元数张量，形状为[*, 4]。与rot_mats参数互斥，只能指定其中一个。
                   如果normalize_quats不为True，则必须是单位四元数
            quats_mask: 四元数掩码张量，形状为[*]。当指定quats且normalize_quats为True时，
                        用于选择哪些四元数元素进行归一化
            normalize_quats: 当指定quats时，是否对四元数进行归一化处理
        """
        # 验证输入参数：必须且只能指定一个旋转表示（旋转矩阵或四元数）
        if (rot_mats is None and quats is None) or (rot_mats is not None and quats is not None):
            raise ValueError("Exactly one input argument must be specified")

        # 验证张量形状：旋转矩阵必须是3x3，四元数必须是4维
        if (rot_mats is not None and rot_mats.shape[-2:] != (3, 3)) or (
            quats is not None and quats.shape[-1] != 4
        ):
            raise ValueError("Incorrectly shaped rotation matrix or quaternion")

        # 强制使用单精度浮点数，确保数值稳定性
        if quats is not None:
            quats = quats.type(torch.float32)
        if rot_mats is not None:
            rot_mats = rot_mats.type(torch.float32)

        # 解析掩码：如果提供了四元数掩码，转换为布尔类型
        if quats is not None and quats_mask is not None:
            quats_mask = quats_mask.type(torch.bool)

        # 四元数归一化处理：确保四元数为单位四元数
        if quats is not None and normalize_quats:
            if quats_mask is not None:
                # 使用掩码选择性地归一化四元数
                quats[quats_mask] = quats[quats_mask] / torch.linalg.norm(
                    quats[quats_mask], dim=-1, keepdim=True
                )
            else:
                # 对所有四元数进行归一化
                quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)

        # 存储旋转表示：可以是旋转矩阵或四元数，但不能同时存储两种
        self._rot_mats = rot_mats
        self._quats = quats

    @staticmethod
    def identity(
        shape: Tuple[Union[int, np.int64]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ):
        """返回单位旋转。

        参数:
            shape: 结果Rotation对象的“形状”
            dtype: 旋转的torch dtype
            device: 新旋转的torch设备
            requires_grad: 是否需要梯度
            fmt: "quat"或"rot_mat"，决定底层格式
        返回:
            新的单位旋转
        """
        if fmt == "rot_mat":
            rot_mats = identity_rot_mats(
                shape,
                dtype,
                device,
                requires_grad,
            )
            return Rotation(rot_mats=rot_mats, quats=None)
        elif fmt == "quat":
            quats = identity_quats(shape, dtype, device, requires_grad)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError(f"Invalid format: f{fmt}")

    # Magic methods

    def __getitem__(self, index: Any):
        """支持torch风格的索引。见shape属性文档。

        参数:
            index: torch索引
        返回:
            索引后的旋转
        """
        if type(index) != tuple:
            index = (index,)

        if self._rot_mats is not None:
            rot_mats = self._rot_mats[index + (slice(None), slice(None))]
            return Rotation(rot_mats=rot_mats)
        elif self._quats is not None:
            quats = self._quats[index + (slice(None),)]
            return Rotation(quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __mul__(self, right: torch.Tensor) -> "Rotation":
        """逐点左乘，可用于掩码Rotation。

        参数:
            right: 张量乘数
        返回:
            乘积
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats * right[..., None, None]
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats * right[..., None]
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def __rmul__(self, left: torch.Tensor) -> "Rotation":
        """逐点右乘。

        参数:
            left: 左乘数
        返回:
            乘积
        """
        return self.__mul__(left)

    # Properties

    @property
    def shape(self) -> torch.Size:
        """返回旋转对象的虚拟形状。即底层旋转矩阵或四元数的批次维度。

        返回:
            虚拟形状
        """
        s = None
        if self._quats is not None:
            s = self._quats.shape[:-1]
        else:
            s = self._rot_mats.shape[:-2]

        return s

    @property
    def dtype(self) -> torch.dtype:
        """返回底层旋转的dtype。"""
        if self._rot_mats is not None:
            return self._rot_mats.dtype
        elif self._quats is not None:
            return self._quats.dtype
        else:
            raise ValueError("Both rotations are None")

    @property
    def device(self) -> torch.device:
        """返回底层旋转的设备。"""
        if self._rot_mats is not None:
            return self._rot_mats.device
        elif self._quats is not None:
            return self._quats.device
        else:
            raise ValueError("Both rotations are None")

    @property
    def requires_grad(self) -> bool:
        """返回底层旋转的requires_grad属性。"""
        if self._rot_mats is not None:
            return self._rot_mats.requires_grad
        elif self._quats is not None:
            return self._quats.requires_grad
        else:
            raise ValueError("Both rotations are None")

    def reshape(
        self,
        new_rots_shape: Optional[torch.Size] = None,
    ) -> "Rotation":
        """返回重塑后的旋转。"""
        if self._quats is not None:
            new_rots = self._quats.reshape(new_rots_shape) if new_rots_shape else self._quats
            new_rot = Rotation(quats=new_rots, normalize_quats=False)
        else:
            new_rots = self._rot_mats.reshape(new_rots_shape) if new_rots_shape else self._rot_mats
            new_rot = Rotation(rot_mats=new_rots, normalize_quats=False)

        return new_rot

    def get_rot_mats(self) -> ROTATION_TENSOR_TYPE:
        """以旋转矩阵张量形式返回底层旋转。"""
        rot_mats = self._rot_mats
        if rot_mats is None:
            if self._quats is None:
                raise ValueError("Both rotations are None")
            else:
                rot_mats = quat_to_rot(self._quats)

        return rot_mats

    def get_quats(self) -> QUATERNION_TENSOR_TYPE:
        """以四元数张量形式返回底层旋转。"""
        quats = self._quats
        if quats is None:
            if self._rot_mats is None:
                raise ValueError("Both rotations are None")
            else:
                quats = rot_to_quat(self._rot_mats)

        return quats

    def get_cur_rot(self) -> Union[QUATERNION_TENSOR_TYPE, ROTATION_TENSOR_TYPE]:
        """返回当前存储的旋转。"""
        if self._rot_mats is not None:
            return self._rot_mats
        elif self._quats is not None:
            return self._quats
        else:
            raise ValueError("Both rotations are None")

    def get_rotvec(self, eps: float = 1e-6) -> torch.Tensor:
        """返回轴-角旋转向量。

        参考scipy实现。
        返回:
            轴-角向量
        """
        quat = self.get_quats()
        # w > 0 to ensure 0 <= angle <= pi
        flip = (quat[..., :1] < 0).float()
        quat = (-1 * quat) * flip + (1 - flip) * quat

        angle = 2 * torch.atan2(torch.linalg.norm(quat[..., 1:], dim=-1), quat[..., 0])

        angle2 = angle * angle
        small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
        large_angle_scales = angle / torch.sin(angle / 2 + eps)

        small_angles = (angle <= 1e-3).float()
        rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
        rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
        return rot_vec

    # Rotation functions

    def compose_q_update_vec(
        self,
        q_update_vec: torch.Tensor,
        normalize_quats: bool = True,
        update_mask: Optional[UPDATE_NODE_MASK_TENSOR_TYPE] = None,
    ) -> "Rotation":
        """用四元数更新向量更新当前旋转，返回新Rotation。

        参数:
            q_update_vec: [*, 3] 四元数更新张量
            normalize_quats: 是否归一化输出四元数
            update_mask: 可选掩码
        返回:
            更新后的Rotation
        """
        quats = self.get_quats()
        quat_update = quat_multiply_by_vec(quats, q_update_vec)
        if update_mask is not None:
            quat_update = quat_update * update_mask
        new_quats = quats + quat_update
        return Rotation(
            rot_mats=None,
            quats=new_quats,
            quats_mask=update_mask.squeeze(-1),
            normalize_quats=normalize_quats,
        )

    def compose_r(self, r: "Rotation") -> "Rotation":
        """将当前Rotation与另一个Rotation的旋转矩阵复合。

        参数:
            r: 更新旋转对象
        返回:
            更新后的旋转对象
        """
        r1 = self.get_rot_mats()
        r2 = r.get_rot_mats()
        new_rot_mats = rot_matmul(r1, r2)
        return Rotation(rot_mats=new_rot_mats, quats=None)

    def compose_q(self, r: "Rotation", normalize_quats: bool = True) -> "Rotation":
        """将当前Rotation与另一个Rotation的四元数复合。

        参数:
            r: 更新旋转对象
        返回:
            更新后的旋转对象
        """
        q1 = self.get_quats()
        q2 = r.get_quats()
        new_quats = quat_multiply(q1, q2)
        return Rotation(rot_mats=None, quats=new_quats, normalize_quats=normalize_quats)

    def apply(self, pts: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        """将当前Rotation作为旋转矩阵应用于一组3D坐标。

        参数:
            pts: [*, 3] 坐标点
        返回:
            旋转后的点
        """
        rot_mats = self.get_rot_mats()
        return rot_vec_mul(rot_mats, pts)

    def invert_apply(self, pts: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        """apply()方法的逆操作。

        参数:
            pts: [*, 3] 坐标点
        返回:
            逆旋转后的点
        """
        rot_mats = self.get_rot_mats()
        inv_rot_mats = invert_rot_mat(rot_mats)
        return rot_vec_mul(inv_rot_mats, pts)

    def invert(self, mask: Optional[NODE_MASK_TENSOR_TYPE] = None) -> "Rotation":
        """返回当前Rotation的逆。

        参数:
            mask: 可选掩码
        返回:
            逆旋转
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=invert_rot_mat(self._rot_mats), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=invert_quat(self._quats, mask=mask),
                normalize_quats=False,
                quats_mask=mask,
            )
        else:
            raise ValueError("Both rotations are None")

    # "Tensor" stuff

    def unsqueeze(self, dim: int) -> "Rotation":
        """类似torch.unsqueeze。维度相对于Rotation对象的shape。

        参数:
            dim: 维度索引
        返回:
            扩展后的Rotation
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        if self._rot_mats is not None:
            rot_mats = self._rot_mats.unsqueeze(dim if dim >= 0 else dim - 2)
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = self._quats.unsqueeze(dim if dim >= 0 else dim - 1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    @staticmethod
    def cat(rs, dim: int) -> "Rotation":
        """沿批次维度拼接旋转。类似torch.cat()。

        注意输出总是旋转矩阵格式。

        参数:
            rs: 旋转对象列表
            dim: 拼接维度
        返回:
            拼接后的Rotation
        """
        rot_mats = [r.get_rot_mats() for r in rs]
        rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)

        return Rotation(rot_mats=rot_mats, quats=None)

    def map_tensor_fn(self, fn: Callable) -> "Rotation":
        """对底层旋转张量应用Tensor->Tensor函数，映射旋转维度。

        参数:
            fn: Tensor->Tensor函数
        返回:
            变换后的Rotation
        """
        if self._rot_mats is not None:
            rot_mats = self._rot_mats.view(self._rot_mats.shape[:-2] + (9,))
            rot_mats = torch.stack(list(map(fn, torch.unbind(rot_mats, dim=-1))), dim=-1)
            rot_mats = rot_mats.view(rot_mats.shape[:-1] + (3, 3))
            return Rotation(rot_mats=rot_mats, quats=None)
        elif self._quats is not None:
            quats = torch.stack(list(map(fn, torch.unbind(self._quats, dim=-1))), dim=-1)
            return Rotation(rot_mats=None, quats=quats, normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def cuda(self) -> "Rotation":
        """类似torch Tensor的cuda()方法。

        返回:
            CUDA内存中的Rotation副本
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)
        elif self._quats is not None:
            return Rotation(rot_mats=None, quats=self._quats.cuda(), normalize_quats=False)
        else:
            raise ValueError("Both rotations are None")

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]) -> "Rotation":
        """类似torch Tensor的to()方法。

        参数:
            device: torch设备
            dtype: torch dtype
        返回:
            使用新设备和dtype的Rotation副本
        """
        if self._rot_mats is not None:
            return Rotation(
                rot_mats=self._rot_mats.to(device=device, dtype=dtype),
                quats=None,
            )
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.to(device=device, dtype=dtype),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")

    def detach(self) -> "Rotation":
        """返回已从torch计算图分离的Rotation副本。

        返回:
            已分离的Rotation副本
        """
        if self._rot_mats is not None:
            return Rotation(rot_mats=self._rot_mats.detach(), quats=None)
        elif self._quats is not None:
            return Rotation(
                rot_mats=None,
                quats=self._quats.detach(),
                normalize_quats=False,
            )
        else:
            raise ValueError("Both rotations are None")


class Rigid:
    """表示刚体变换的类。

    本质上是对两个对象的封装：一个Rotation对象和一个[*, 3]平移张量。
    设计为近似于单个torch张量的行为，具有其组成部分共享批次维度的形状。
    """

    def __init__(
        self,
        rots: Optional[Rotation],
        trans: Optional[COORDINATES_TENSOR_TYPE],
    ):
        """
        参数:
            rots: [*, 3, 3] 旋转张量
            trans: [*, 3] 平移张量
        """
        # (we need device, dtype, etc. from at least one input)

        batch_dims, dtype, device, requires_grad = None, None, None, None
        if trans is not None:
            batch_dims = trans.shape[:-1]
            dtype = trans.dtype
            device = trans.device
            requires_grad = trans.requires_grad
        elif rots is not None:
            batch_dims = rots.shape
            dtype = rots.dtype
            device = rots.device
            requires_grad = rots.requires_grad
        else:
            raise ValueError("At least one input argument must be specified")

        if rots is None:
            rots = Rotation.identity(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )
        elif trans is None:
            trans = identity_trans(
                batch_dims,
                dtype,
                device,
                requires_grad,
            )

        if (rots.shape != trans.shape[:-1]) or (rots.device != trans.device):
            raise ValueError("Rots and trans incompatible")

        # Force full precision. Happens to the rotations automatically.
        trans = trans.type(torch.float32)

        self._rots = rots
        self._trans = trans

    @staticmethod
    def identity(
        shape: Tuple[Union[int, np.int64]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = True,
        fmt: str = "quat",
    ) -> "Rigid":
        """构造单位变换。

        参数:
            shape: 期望形状
            dtype: 内部张量的dtype
            device: 内部张量的设备
            requires_grad: 是否启用梯度
        返回:
            单位变换
        """
        return Rigid(
            Rotation.identity(shape, dtype, device, requires_grad, fmt=fmt),
            identity_trans(shape, dtype, device, requires_grad),
        )

    def __getitem__(self, index: Any) -> "Rigid":
        """用PyTorch风格索引刚体变换。索引应用于旋转和平移的共享维度。

        例如::

            r = Rotation(rot_mats=torch.rand(10, 10, 3, 3), quats=None)
            t = Rigid(r, torch.rand(10, 10, 3))
            indexed = t[3, 4:6]
            assert(indexed.shape == (2,))
            assert(indexed.get_rots().shape == (2,))
            assert(indexed.get_trans().shape == (2, 3))

        参数:
            index: 标准torch张量索引
        返回:
            索引后的张量
        """
        if type(index) != tuple:
            index = (index,)

        return Rigid(
            self._rots[index],
            self._trans[index + (slice(None),)],
        )

    def __mul__(self, right: torch.Tensor) -> "Rigid":
        """逐点左乘，可用于掩码Rigid。

        参数:
            right: 张量乘数
        返回:
            乘积
        """
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._rots * right
        new_trans = self._trans * right[..., None]

        return Rigid(new_rots, new_trans)

    def __rmul__(self, left: torch.Tensor) -> "Rigid":
        """逐点右乘。

        参数:
            left: 左乘数
        返回:
            乘积
        """
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """返回旋转和平移的共享维度的形状。

        返回:
            变换的形状
        """
        s = self._trans.shape[:-1]
        return s

    @property
    def device(self) -> torch.device:
        """返回Rigid张量所在的设备。

        返回:
            张量所在的设备
        """
        return self._trans.device

    def reshape(
        self,
        new_rots_shape: Optional[torch.Size] = None,
        new_trans_shape: Optional[torch.Size] = None,
    ) -> "Rigid":
        """返回重塑后的旋转和平移。

        返回:
            重塑后的变换
        """
        new_rots = (
            self._rots.reshape(new_rots_shape=new_rots_shape) if new_rots_shape else self._rots
        )
        new_trans = self._trans.reshape(new_trans_shape) if new_trans_shape else self._trans

        return Rigid(new_rots, new_trans)

    def get_rots(self) -> Rotation:
        """获取旋转对象。

        返回:
            旋转对象
        """
        return self._rots

    def get_trans(self) -> COORDINATES_TENSOR_TYPE:
        """获取平移。

        返回:
            存储的平移
        """
        return self._trans

    def compose_q_update_vec(
        self,
        q_update_vec: Float[torch.Tensor, "... num_nodes 6"],  # noqa: F722
        update_mask: Optional[UPDATE_NODE_MASK_TENSOR_TYPE] = None,
    ) -> "Rigid":
        """用形状为[*, 6]的四元数更新向量复合变换。

        参数:
            q_update_vec: 四元数更新向量
            update_mask: 可选掩码
        返回:
            复合变换
        """
        q_vec, t_vec = q_update_vec[..., :3], q_update_vec[..., 3:]
        new_rots = self._rots.compose_q_update_vec(q_vec, update_mask=update_mask)

        trans_update = self._rots.apply(t_vec)
        if update_mask is not None:
            trans_update = trans_update * update_mask
        new_translation = self._trans + trans_update

        return Rigid(new_rots, new_translation)

    def compose(self, r: "Rigid") -> "Rigid":
        """将当前刚体对象与另一个刚体对象复合。

        参数:
            r: 另一个Rigid对象
        返回:
            两个变换的复合
        """
        new_rot = self._rots.compose_r(r._rots)
        new_trans = self._rots.apply(r._trans) + self._trans
        return Rigid(new_rot, new_trans)

    def compose_r(self, rot: "Rigid", order: str = "right") -> "Rigid":
        """将当前刚体对象与另一个刚体对象复合。

        参数:
            r: 另一个Rigid对象
            order: 旋转乘法顺序
        返回:
            两个变换的复合
        """
        if order == "right":
            new_rot = self._rots.compose_r(rot)
        elif order == "left":
            new_rot = rot.compose_r(self._rots)
        else:
            raise ValueError(f"Unrecognized multiplication order: {order}")
        return Rigid(new_rot, self._trans)

    def apply(self, pts: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        """将变换应用于坐标张量。

        参数:
            pts: [*, 3] 坐标张量
        返回:
            变换后的点
        """
        rotated = self._rots.apply(pts)
        return rotated + self._trans

    def invert_apply(self, pts: COORDINATES_TENSOR_TYPE) -> COORDINATES_TENSOR_TYPE:
        """将变换的逆应用于坐标张量。

        参数:
            pts: [*, 3] 坐标张量
        返回:
            变换后的点
        """
        pts = pts - self._trans
        return self._rots.invert_apply(pts)

    def invert(self) -> "Rigid":
        """求变换的逆。

        返回:
            逆变换
        """
        rot_inv = self._rots.invert()
        trn_inv = rot_inv.apply(self._trans)

        return Rigid(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable) -> "Rigid":
        """对底层平移和旋转张量应用Tensor->Tensor函数，分别映射平移/旋转维度。

        参数:
            fn: Tensor->Tensor函数
        返回:
            变换后的Rigid对象
        """
        new_rots = self._rots.map_tensor_fn(fn)
        new_trans = torch.stack(list(map(fn, torch.unbind(self._trans, dim=-1))), dim=-1)

        return Rigid(new_rots, new_trans)

    def to_tensor_4x4(self) -> Float[torch.Tensor, "... num_nodes 4 4"]:  # noqa: F722
        """将变换转换为齐次变换张量。

        返回:
            [*, 4, 4] 齐次变换张量
        """
        tensor = self._trans.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._rots.get_rot_mats()
        tensor[..., :3, 3] = self._trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(t: Float[torch.Tensor, "... num_nodes 4 4"]) -> "Rigid":  # noqa: F722
        """从齐次变换张量构造变换。

        参数:
            t: [*, 4, 4] 齐次变换张量
        返回:
            形状为[*]的T对象
        """
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")

        rots = Rotation(rot_mats=t[..., :3, :3], quats=None)
        trans = t[..., :3, 3]

        return Rigid(rots, trans)

    def to_tensor_7(self) -> Float[torch.Tensor, "... num_nodes 7"]:  # noqa: F722
        """将变换转换为7列张量，前四列为四元数，后三列为平移。

        返回:
            [*, 7] 变换张量
        """
        tensor = self._trans.new_zeros((*self.shape, 7))
        tensor[..., :4] = self._rots.get_quats()
        tensor[..., 4:] = self._trans

        return tensor

    @staticmethod
    def from_tensor_7(
        t: Float[torch.Tensor, "... num_nodes 7"], normalize_quats: bool = False  # noqa: F722
    ) -> "Rigid":
        """从7列张量构造变换。"""
        if t.shape[-1] != 7:
            raise ValueError("Incorrectly shaped input tensor")

        quats, trans = t[..., :4], t[..., 4:]

        rots = Rotation(rot_mats=None, quats=quats, normalize_quats=normalize_quats)

        return Rigid(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: COORDINATES_TENSOR_TYPE,
        origin: COORDINATES_TENSOR_TYPE,
        p_xy_plane: COORDINATES_TENSOR_TYPE,
        eps: float = 1e-8,
    ) -> "Rigid":
        """实现算法21。用Gram-Schmidt算法从3个点构造变换。

        参数:
            p_neg_x_axis: [*, 3] 坐标
            origin: [*, 3] 坐标，作为框架原点
            p_xy_plane: [*, 3] 坐标
            eps: 小的epsilon值
        返回:
            形状为[*]的变换对象
        """
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum(c * c for c in e0) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum(c * c for c in e1) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, torch.stack(origin, dim=-1))

    def unsqueeze(self, dim: int) -> "Rigid":
        """类似torch.unsqueeze。维度相对于旋转/平移的共享维度。

        参数:
            dim: 维度索引
        返回:
            扩展后的变换
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._rots.unsqueeze(dim)
        trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    @staticmethod
    def cat(ts: List["Rigid"], dim: int) -> "Rigid":
        """沿新维度拼接变换。

        参数:
            ts: T对象列表
            dim: 拼接维度
        返回:
            拼接后的变换对象
        """
        rots = Rotation.cat([t._rots for t in ts], dim)
        trans = torch.cat([t._trans for t in ts], dim=dim if dim >= 0 else dim - 1)

        return Rigid(rots, trans)

    def apply_rot_fn(self, fn: Callable) -> "Rigid":
        """对存储的旋转对象应用Rotation->Rotation函数。

        参数:
            fn: Rotation->Rotation函数
        返回:
            旋转变换后的对象
        """
        return Rigid(fn(self._rots), self._trans)

    def apply_trans_fn(self, fn: Callable) -> "Rigid":
        """对存储的平移应用Tensor->Tensor函数。

        参数:
            fn: Tensor->Tensor函数
        返回:
            平移变换后的对象
        """
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float) -> "Rigid":
        """按常数因子缩放平移。

        参数:
            trans_scale_factor: 常数因子
        返回:
            平移缩放后的对象
        """
        return self.apply_trans_fn(lambda t: t * trans_scale_factor)

    def stop_rot_gradient(self) -> "Rigid":
        """分离底层旋转对象的梯度。

        返回:
            旋转已分离的对象
        """
        return self.apply_rot_fn(lambda r: r.detach())

    @staticmethod
    def make_transform_from_reference(
        n_xyz: COORDINATES_TENSOR_TYPE,
        ca_xyz: COORDINATES_TENSOR_TYPE,
        c_xyz: COORDINATES_TENSOR_TYPE,
        eps: float = 1e-20,
    ) -> "Rigid":
        """返回变换对象从参考坐标。

        注意此方法不处理对称性。若原子顺序不标准，N原子不会在标准位置。
        需自行处理此类情况。

        参数:
            n_xyz: [*, 3] 氮原子xyz坐标
            ca_xyz: [*, 3] 碳α原子xyz坐标
            c_xyz: [*, 3] 碳原子xyz坐标
        返回:
            变换对象。应用变换后，参考主链坐标将近似等于输入坐标。
        """
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = (c_xyz[..., i] for i in range(3))
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm
        zeros = sin_c1.new_zeros(sin_c1.shape)
        ones = sin_c1.new_ones(sin_c1.shape)

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x**2 + c_y**2 + c_z**2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c1_rots[..., 2, 0] = -1 * sin_c2
        c1_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = (n_xyz[..., i] for i in range(3))
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, translation)

    def cuda(self) -> "Rigid":
        """将变换对象移动到GPU内存。

        返回:
            GPU上的变换对象
        """
        return Rigid(self._rots.cuda(), self._trans.cuda())
            A transformation object with a transformed translation.
        """
        return Rigid(self._rots, fn(self._trans))

    def scale_translation(self, trans_scale_factor: float) -> "Rigid":
        """Scales the translation by a constant factor.

        Args:
            trans_scale_factor:
                The constant factor
        Returns:
            A transformation object with a scaled translation.
        """
        return self.apply_trans_fn(lambda t: t * trans_scale_factor)

    def stop_rot_gradient(self) -> "Rigid":
        """Detaches the underlying rotation object.

        Returns:
            A transformation object with detached rotations
        """
        return self.apply_rot_fn(lambda r: r.detach())

    @staticmethod
    def make_transform_from_reference(
        n_xyz: COORDINATES_TENSOR_TYPE,
        ca_xyz: COORDINATES_TENSOR_TYPE,
        c_xyz: COORDINATES_TENSOR_TYPE,
        eps: float = 1e-20,
    ) -> "Rigid":
        """Returns a transformation object from reference coordinates.

        Note that this method does not take care of symmetries. If you
        provide the atom positions in the non-standard way, the N atom will
        end up not at [-0.527250, 1.359329, 0.0] but instead at
        [-0.527250, -1.359329, 0.0]. You need to take care of such cases in
        your code.

        Args:
            n_xyz: A [*, 3] tensor of nitrogen xyz coordinates.
            ca_xyz: A [*, 3] tensor of carbon alpha xyz coordinates.
            c_xyz: A [*, 3] tensor of carbon xyz coordinates.
        Returns:
            A transformation object. After applying the translation and
            rotation to the reference backbone, the coordinates will
            approximately equal to the input coordinates.
        """
        translation = -1 * ca_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = (c_xyz[..., i] for i in range(3))
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm
        zeros = sin_c1.new_zeros(sin_c1.shape)
        ones = sin_c1.new_ones(sin_c1.shape)

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x**2 + c_y**2 + c_z**2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c1_rots[..., 2, 0] = -1 * sin_c2
        c1_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = (n_xyz[..., i] for i in range(3))
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        rot_obj = Rotation(rot_mats=rots, quats=None)

        return Rigid(rot_obj, translation)

    def cuda(self) -> "Rigid":
        """Moves the transformation object to GPU memory.

        Returns:
            A version of the transformation on GPU
        """
        return Rigid(self._rots.cuda(), self._trans.cuda())
