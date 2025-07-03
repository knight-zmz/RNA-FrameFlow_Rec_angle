"""
用于计算全原子表示的工具函数。

代码改编自
- https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/all_atom.py
- https://github.com/Profluent-Internships/MMDiff/blob/main/src/data/components/pdb/all_atom.py
"""

import torch
import torch.nn.functional as F

'''from rna_backbone_design.data import nucleotide_constants
from rna_backbone_design.data import rigid_utils as ru
from rna_backbone_design.data import vocabulary
from rna_backbone_design.data.complex_constants import NUM_NA_TORSIONS, NUM_PROT_NA_TORSIONS'''

# 使用相对导入，因为所有模块都在同一个data包内
from nucleotide_constants import *
from rigid_utils import Rigid, Rotation
from vocabulary import *
from complex_constants import NUM_NA_TORSIONS, NUM_PROT_NA_TORSIONS

# esm3 官方提供的函数库
from esm.utils.structure.affine3d import (
    Affine3D,
    build_affine3d_from_coordinates,
)

IDEALIZED_NA_ATOM_POS27 = torch.tensor(nttype_atom27_rigid_group_positions)
IDEALIZED_NA_ATOM_POS27_MASK = torch.any(IDEALIZED_NA_ATOM_POS27, axis=-1)
IDEALIZED_NA_ATOM_POS23 = torch.tensor(
    nttype_compact_atom_rigid_group_positions
)
DEFAULT_NA_RESIDUE_FRAMES = torch.tensor(nttype_rigid_group_default_frame)
NA_RESIDUE_ATOM23_MASK = torch.tensor(nttype_compact_atom_mask)
NA_RESIDUE_GROUP_IDX = torch.tensor(nttype_compact_atom_to_rigid_group)

def create_rna_rigid(rots, trans):
    # 输入分开的旋转和平移，返回统一的 rigid_utils.Rigid(...) 对象
    rots = Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)

def to_atom37_rna(trans, rots, is_na_residue_mask, torsions=None):
    """
    参数:
        trans (tensor): 表示平移的张量
        rots (tensor): 表示旋转的张量
        is_na_residue_mask (tensor): 单热核苷酸掩码
        torsions (tensor): 预测/真实的扭转角，用于补全非框架骨架原子

    说明:
        将RNA的旋转+平移（即RNA框架）转换为稀疏的ATOM37表示
    """

    final_atom37 = compute_backbone(
        bb_rigids=create_rna_rigid(rots, trans),
        torsions=torsions, # 16 float values -> NUM_NA_TORSIONS * 2 (each angle is in SO(2))
        is_na_residue_mask=is_na_residue_mask,
    )[0]
    return final_atom37



#########################################################
## 定义函数：【将刚体变换轨迹、扭转角、核苷酸掩码】转换为ATOM37格式的RNA全原子结构
#########################################################
def transrot_to_atom37_rna(transrot_traj, is_na_residue_mask, torsions):
    """
    将刚体变换轨迹转换为ATOM37格式的RNA全原子结构

    参数:
        transrot_traj (list): 刚体变换轨迹列表，包含N_T个流匹配时间步的演化刚体变换
                              每个元素是(trans, rots)元组，表示平移和旋转
        is_na_residue_mask (tensor): 核苷酸掩码，标记哪些位置是RNA残基
        torsions (tensor): 从AngleResNet预测的扭转角，形状为16（NUM_NA_TORSIONS * 2）

    返回:
        atom37_traj (list): 全原子骨架轨迹列表，包含所有非框架原子根据预测扭转角放置的坐标
    """
    # 初始化ATOM37轨迹列表
    atom37_traj = []

    # 遍历刚体变换轨迹中的每个时间步
    for trans, rots in transrot_traj:
        # 从旋转矩阵和平移向量创建RNA刚体对象
        rigids = create_rna_rigid(rots, trans)
        
        # 计算全原子骨架结构
        # 使用刚体变换、扭转角和RNA掩码生成完整的RNA原子坐标
        rna_atom37 = compute_backbone(
                bb_rigids=rigids,  # 骨架刚体变换
                torsions=torsions,  # 预测的扭转角（16个角度）
                is_na_residue_mask=is_na_residue_mask,  # RNA残基掩码
            )[0]  # 取第一个返回值（原子坐标）
        
        # 将结果分离到CPU并添加到轨迹中
        rna_atom37 = rna_atom37.detach().cpu()  # 分离梯度并移到CPU
        atom37_traj.append(rna_atom37)  # 添加到轨迹列表

    return atom37_traj
#########################################################







#########################################################
## 定义函数：【将刚体变换轨迹、扭转角、核苷酸掩码】转换为ATOM23格式的RNA全原子结构
#########################################################
def transrot_to_atom23_rna(transrot_traj, is_na_residue_mask, torsions):
    """
    将刚体变换轨迹转换为ATOM23格式的RNA全原子结构

    参数:
        transrot_traj (list): 刚体变换轨迹列表，包含N_T个流匹配时间步的演化刚体变换
                              每个元素是(trans, rots)元组，表示平移和旋转
        is_na_residue_mask (tensor): 核苷酸掩码，标记哪些位置是RNA残基
        torsions (tensor): 从AngleResNet预测的扭转角，形状为16（NUM_NA_TORSIONS * 2）

    返回:
        atom23_traj (list): 全原子骨架轨迹列表，包含所有非框架原子根据预测扭转角放置的坐标
    """
    # 初始化ATOM23轨迹列表
    atom23_traj = []

    # 遍历刚体变换轨迹中的每个时间步
    for trans, rots in transrot_traj:
        # 从旋转矩阵和平移向量创建RNA刚体对象
        rigids = create_rna_rigid(rots, trans)
        
        # 计算全原子骨架结构
        # 使用刚体变换、扭转角和RNA掩码生成完整的RNA原子坐标
        rna_atom23 = compute_backbone(
                bb_rigids=rigids,  # 骨架刚体变换
                torsions=torsions,  # 预测的扭转角（16个角度）
                is_na_residue_mask=is_na_residue_mask,  # RNA残基掩码
            )
        
        # 将结果分离到CPU并添加到轨迹中
        rna_atom23 = rna_atom23.detach().cpu()  # 分离梯度并移到CPU
        atom23_traj.append(rna_atom23)  # 添加到轨迹列表

    return atom23_traj
#########################################################






#########################################################
def of_na_torsion_angles_to_frames(r, alpha, aatype, rrgdf):
    """
    将RNA扭转角转换为刚体框架

    参数:
        r (tensor): 表示RNA框架的rigid_utils.Rigid(...)对象张量
        alpha (tensor): 从AngleResNet预测的扭转角
        aatype (tensor): 残基类型张量 [忽略]
        rrgdf (tensor): 默认的RNA残基刚体组框架张量

    说明:
        使用预测的RNA框架`r`和提供的扭转角`alpha`
        将非框架骨架原子插值到ATOM37表示中。

    返回:
        包含所有13个RNA骨架原子的ATOM37格式的torch张量。
    """
    # 获取默认的4x4变换矩阵 [*, N, 11, 4, 4]
    # 根据残基类型选择对应的默认刚体组框架
    default_4x4 = rrgdf[aatype, ...]

    # 将4x4矩阵转换为刚体变换对象 [*, N, 11]
    # 包含一个[*, N, 11, 3, 3]旋转矩阵和一个[*, N, 11, 3]平移矩阵
    default_r = r.from_tensor_4x4(default_4x4)

    # 创建第一个骨架旋转的占位符
    # 初始化为零张量，形状为(*alpha.shape[:-1], 2)
    bb_rot1 = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot1[..., 1] = 1  # 设置第二个元素为1，表示单位旋转

    # 将第一个骨架旋转与预测的扭转角连接
    # 扩展bb_rot1以匹配alpha的形状，然后连接
    alpha = torch.cat(
        [
            bb_rot1.expand(*alpha.shape[:-2], -1, -1),  # 扩展第一个骨架旋转
            alpha,  # 添加预测的扭转角
        ],
        dim=-2,  # 在倒数第二个维度连接
    )

    # 创建旋转矩阵 [*, N, 11, 3, 3]
    # 生成如下形式的旋转矩阵：
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # 其中a_1和a_2是扭转角的正弦和余弦值

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1  # 3D旋转矩阵的左上角对角线值
    all_rots[..., 1, 1] = alpha[..., 1]  # 第一个扭转角的正弦值
    all_rots[..., 1, 2] = -alpha[..., 0]  # 第一个扭转角的余弦值（负号）
    all_rots[..., 2, 1:] = alpha  # 剩余扭转角的正弦和余弦值

    # 将旋转矩阵转换为刚体对象（只包含旋转，无平移）
    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    # 将默认框架与扭转角旋转组合，得到所有框架
    all_frames = default_r.compose(all_rots)

    # 提取各个骨架原子的框架
    backbone2_atom1_frame = all_frames[..., 1]
    backbone2_atom2_frame = all_frames[..., 2]
    backbone2_atom3_frame = all_frames[..., 3]
    delta_frame_to_frame = all_frames[..., 4]
    gamma_frame_to_frame = all_frames[..., 5]
    beta_frame_to_frame = all_frames[..., 6]
    alpha1_frame_to_frame = all_frames[..., 7]
    alpha2_frame_to_frame = all_frames[..., 8]
    tm_frame_to_frame = all_frames[..., 9]
    chi_frame_to_frame = all_frames[..., 10]

    # 将各个框架转换到骨架坐标系
    backbone2_atom1_frame_to_bb = backbone2_atom1_frame  # C2'直接使用
    backbone2_atom2_frame_to_bb = backbone2_atom2_frame  # C1'直接使用
    # 注意：N9/N1基于C1'的相对位置构建
    backbone2_atom3_frame_to_bb = backbone2_atom2_frame.compose(backbone2_atom3_frame)
    delta_frame_to_bb = delta_frame_to_frame  # O3'直接使用
    gamma_frame_to_bb = gamma_frame_to_frame  # O5'直接使用
    beta_frame_to_bb = gamma_frame_to_bb.compose(beta_frame_to_frame)  # P相对于O5'
    alpha1_frame_to_bb = beta_frame_to_bb.compose(alpha1_frame_to_frame)  # OP1相对于P
    alpha2_frame_to_bb = beta_frame_to_bb.compose(alpha2_frame_to_frame)  # OP2相对于P
    # 使用`backbone2_atom1/3_frames`来组合`tm`和`chi`框架，
    # 因为`backbone2_atom1/3_frames`定位了`C2'`和`N9/N1`原子
    # （即第二骨架组的两个原子）
    tm_frame_to_bb = backbone2_atom1_frame_to_bb.compose(tm_frame_to_frame)  # O2'相对于C2'
    chi_frame_to_bb = backbone2_atom3_frame_to_bb.compose(chi_frame_to_frame)  # 碱基相对于N9/N1

    # 将所有框架连接成一个张量，形状为[*, N, 11]
    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., 0].unsqueeze(-1),  # 第一骨架组
            backbone2_atom1_frame_to_bb.unsqueeze(-1),  # C2'框架
            backbone2_atom2_frame_to_bb.unsqueeze(-1),  # C1'框架
            backbone2_atom3_frame_to_bb.unsqueeze(-1),  # N9/N1框架
            delta_frame_to_bb.unsqueeze(-1),  # O3'框架
            gamma_frame_to_bb.unsqueeze(-1),  # O5'框架
            beta_frame_to_bb.unsqueeze(-1),  # P框架
            alpha1_frame_to_bb.unsqueeze(-1),  # OP1框架
            alpha2_frame_to_bb.unsqueeze(-1),  # OP2框架
            tm_frame_to_bb.unsqueeze(-1),  # O2'框架
            chi_frame_to_bb.unsqueeze(-1),  # 碱基框架
        ],
        dim=-1,  # 在最后一个维度连接
    )

    # 将所有框架转换到全局坐标系
    # 通过将RNA框架与各个骨架框架组合
    all_frames_to_global = r[..., None].compose(all_frames_to_bb)
    return all_frames_to_global
#########################################################






#########################################################
def na_frames_to_atom23_pos(r, aatype):
    """
    将核酸（NA）框架转换为理想化的全原子表示

    参数:
        r: 所有刚体组框架 [..., N, 11, 3]
        aatype: 残基类型 [..., N]

    说明:
        将核酸（NA）框架转换为其理想化的全原子表示

    返回:
        以ATOM37格式存储的理想化全原子骨架的torch张量
    """

    # 获取每个原子所属的刚体组索引 [*, N, 23]
    # 根据残基类型和原子类型确定每个原子属于哪个刚体组（0-10）
    group_mask = NA_RESIDUE_GROUP_IDX.to(r.device)[aatype, ...]

    # 将组索引转换为one-hot编码 [*, N, 23, 11]
    # 每个原子位置都有一个11维的one-hot向量，表示它属于哪个刚体组
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_NA_RESIDUE_FRAMES.shape[-3],  # 11个刚体组
    ).to(r.device)

    # 将刚体框架与one-hot掩码相乘 [*, N, 23, 11]
    # 为每个原子选择对应的刚体变换框架
    t_atoms_to_global = r[..., None, :] * group_mask

    # 对最后一个维度求和，得到每个原子的变换 [*, N, 23]
    # 将one-hot编码的多个框架合并为单个变换
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # 创建原子掩码，标记哪些原子位置是有效的 [*, N, 23, 1]
    # 某些残基类型可能没有特定的原子，用掩码标记
    frame_atom_mask = NA_RESIDUE_ATOM23_MASK.to(r.device)[aatype, ...].unsqueeze(-1)

    # 获取理想化的原子位置 [*, N, 23, 3]
    # 这些是相对于刚体组的理想化原子坐标
    frame_null_pos = IDEALIZED_NA_ATOM_POS23.to(r.device)[aatype, ...]
    
    # 将理想化位置通过刚体变换转换到全局坐标系
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    
    # 应用原子掩码，将无效原子位置置零
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions
#########################################################






#########################################################
## 计算全原子骨架结构
#########################################################
def compute_backbone(bb_rigids, torsions, is_na_residue_mask, aatype=None):
    """
    计算RNA全原子骨架结构

    参数:
        bb_rigids (tensor): 包含旋转和平移的刚体变换张量
        torsions: 从AngleResNet预测的扭转角
        is_na_residue_mask (tensor): 核苷酸掩码，标记哪些位置是RNA残基
        aatype (tensor): 残基类型张量 [可选]

    说明:
        此方法接收刚体变换对象并将其转换为全原子RNA骨架结构

    返回:
        包含框架原子和根据扭转角插值的非框架原子的全原子RNA骨架
        以ATOM37格式存储为torch张量
    """

    # 检查是否存在RNA输入
    na_inputs_present = is_na_residue_mask.any().item()
    
    # 重塑扭转角张量：将16个值重新组织为8个扭转角（每个角度用2个值表示）
    torsions = torsions.view(torsions.shape[0], torsions.shape[1], NUM_NA_TORSIONS*2)
    
    # 创建扭转角张量，初始化为前两个扭转角值的平铺
    # 形状扩展为包含所有蛋白质-核酸扭转角位置
    torsion_angles = torch.tile(
        torsions[..., None, :2],  # 取前两个扭转角值
        tuple([1 for _ in range(len(bb_rigids.shape))]) + (NUM_PROT_NA_TORSIONS, 1),
    )

    if na_inputs_present:
        """
        注意：
        对于核酸分子，我们将预测的扭转角插入前8个扭转角条目中，
        并使用前8个预测扭转角中的第一个来平铺剩余的扭转角条目

        尽管data_transforms.py中的atom27_to_frames()方法使用NUM_PROT_NA_TORSIONS，
        但RNA骨架的8个扭转角通过上面的数组切片从其中提取：
        """
        # 将扭转角重新组织为8个角度，每个角度2个值
        masked_angles = torsions.view(torsions.shape[0], -1, 8, 2)
        # 将RNA的扭转角插入到前NUM_NA_TORSIONS个位置
        torsion_angles[..., :NUM_NA_TORSIONS, :] = masked_angles
    
    # 确定旋转维度：如果是四元数则为4，否则为3x3矩阵
    rot_dim = ((4,) if bb_rigids._rots._quats is not None else (3, 3))
    
    # 为RNA填充残基类型 [ACGU]，如果未提供则默认为0
    aatype = (aatype if aatype is not None else torch.zeros(bb_rigids.shape, device=bb_rigids.device, dtype=torch.long))


    if na_inputs_present:

        # 提取RNA残基的刚体变换，重塑为适当的形状
        na_bb_rigids = bb_rigids[is_na_residue_mask].reshape(
            new_rots_shape=torch.Size((bb_rigids.shape[0], -1, *rot_dim)),
            new_trans_shape=torch.Size((bb_rigids.shape[0], -1, 3)),
        )
        
        # 重塑扭转角张量以匹配刚体变换的形状
        na_torsion_angles = torsion_angles.view(torsion_angles.shape[0], -1, NUM_PROT_NA_TORSIONS, 2)

        # 提取RNA残基类型并调整索引范围
        na_aatype = aatype[is_na_residue_mask].view(aatype.shape[0], -1)
        na_aatype_in_original_range = na_aatype.min() > protein_restype_num
        
        # 计算有效的RNA残基类型索引
        effective_na_aatype = (
            na_aatype - (protein_restype_num + 1)
            if na_aatype_in_original_range
            else na_aatype
        )



        #########################################################
        # 使用扭转角将RNA刚体变换转换为所有框架
        all_na_frames = of_na_torsion_angles_to_frames(
            r=na_bb_rigids,  # RNA刚体变换
            alpha=na_torsion_angles,  # 扭转角
            aatype=effective_na_aatype,  # 有效RNA残基类型
            rrgdf=DEFAULT_NA_RESIDUE_FRAMES.to(bb_rigids.device),  # 默认RNA残基框架
        )

        # 将RNA框架转换为23个原子的理想化位置
        na_atom23_pos = na_frames_to_atom23_pos(all_na_frames, effective_na_aatype)
        #########################################################


    if na_inputs_present:
        atom23_pos = na_atom23_pos
    else:
        raise Exception("必须提供蛋白质或核酸链作为输入。")
    
    return atom23_pos
    
    # 初始化ATOM37格式的全原子位置张量
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3), device=bb_rigids.device)
    # 初始化监督掩码，标记哪些原子位置是有效的
    atom37_bb_supervised_mask = torch.zeros(bb_rigids.shape + (37,), device=bb_rigids.device, dtype=torch.bool)

    # 注意：将非框架原子映射到ATOM37表示中的正确索引
    if na_inputs_present:
        # 注意：核酸的atom23骨架顺序 = ['C3'', 'C4'', 'O4'', 'C2'', 'C1'', 'C5'', 'O3'', 'O5'', 'P', 'OP1', 'OP2', 'N9', ...]
        # 注意：核酸的atom37骨架顺序 = ['C1'', 'C2'', 'C3'', 'C4'', 'C5'', '05'', '04'', 'O3'', 'O2'', 'P', 'OP1', 'OP2', 'N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9', ...]
        #                                                         2      3      4             6      7            9    10     11     12
        
        # 将atom23位置重新索引到atom37格式
        atom37_bb_pos[..., 2:4, :][is_na_residue_mask] = atom23_pos[..., :2, :][is_na_residue_mask]  # 重新索引C3'和C4'
        atom37_bb_pos[..., 6, :][is_na_residue_mask] = atom23_pos[..., 2, :][is_na_residue_mask]  # 重新索引O4'
        atom37_bb_pos[..., 1, :][is_na_residue_mask] = atom23_pos[..., 3, :][is_na_residue_mask]  # 重新索引C2'
        atom37_bb_pos[..., 0, :][is_na_residue_mask] = atom23_pos[..., 4, :][is_na_residue_mask]  # 重新索引C1'
        atom37_bb_pos[..., 4, :][is_na_residue_mask] = atom23_pos[..., 5, :][is_na_residue_mask]  # 重新索引C5'
        atom37_bb_pos[..., 7, :][is_na_residue_mask] = atom23_pos[..., 6, :][is_na_residue_mask]  # 重新索引O3'
        atom37_bb_pos[..., 5, :][is_na_residue_mask] = atom23_pos[..., 7, :][is_na_residue_mask]  # 重新索引O5'
        atom37_bb_pos[..., 9:12, :][is_na_residue_mask] = atom23_pos[..., 8:11, :][is_na_residue_mask]  # 重新索引P, OP1, 和OP2
        atom37_bb_pos[..., 18, :][is_na_residue_mask] = atom23_pos[..., 11, :][is_na_residue_mask]  # 重新索引N9，对于嘧啶残基则重新索引为N1
        
        # 设置监督掩码，标记哪些原子位置是有效的
        atom37_bb_supervised_mask[..., :8][is_na_residue_mask] = True  # 前8个骨架原子
        atom37_bb_supervised_mask[..., 9:12][is_na_residue_mask] = True  # 磷酸基团原子
        atom37_bb_supervised_mask[..., 18][is_na_residue_mask] = True  # 碱基原子

    # 创建原子掩码，标记哪些原子位置有非零坐标
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)

    return atom37_bb_pos, atom37_mask, atom37_bb_supervised_mask, aatype, atom23_pos




#########################################################
## 测试
if __name__ == "__main__":

    # 定义批次大小
    bs = 1

    # 定义残基个数
    res_num = 7

    # 定义轨迹数
    traj_num = 2

    # 创建一个简单的刚体变换
    coords = torch.randn(bs, res_num, 3, 3)
    rig = Rigid.from_3_points(
        coords[..., 2, :],
        coords[..., 1, :],
        coords[..., 0, :],
    )
    trans = rig.get_trans()
    rots_mats = rig.get_rots().get_rot_mats()

    # 使用 esm3 官方提供的函数库，构建刚体变换
    '''affine, affine_mask = build_affine3d_from_coordinates(coords)
    esm_trans = affine.trans
    esm_rots_mats = affine.rot.tensor

    print(f'trans: {trans.shape}')
    print(f'trans: {trans[0, 0, :]}')
    print(f'rots_mats: {rots_mats.shape}')
    print(f'rots_mats: {rots_mats[0, 0, :]}')
    print(f'esm_trans: {esm_trans.shape}')
    print(f'esm_trans: {esm_trans[0, 0, :]}')
    print(f'esm_rots_mats: {esm_rots_mats.shape}')
    print(f'esm_rots_mats: {esm_rots_mats[0, 0, :]}')
    assert False'''

    # 测试compute_backbone函数
    # 创建一个简单的刚体变换轨迹
    transrot_traj = [
        (trans, rots_mats) for _ in range(traj_num)
    ]
    # 创建一个简单的扭转角张量
    torsions = torch.randn(bs, res_num, 16)
    # 创建一个简单的核苷酸掩码
    is_na_residue_mask = torch.ones([bs, res_num], dtype=torch.bool)
    #is_na_residue_mask[:, -1] = 0
    
    # 调用 transrot_to_atom23_rna 函数
    atom23_traj = transrot_to_atom23_rna(transrot_traj, is_na_residue_mask, torsions)
    for it in atom23_traj:
        print(f'it: {it.shape}')
