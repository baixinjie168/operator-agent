错误1：
错误路径:  parameter_constraints[3].constraints.shape[0].constraint[0].dim_num
错误信息:  weightOptional的shape维度数量约束与文档不一致。文档明确说明weightOptional的shape为[H]，即维度数量固定为1，但JSON中dim_num为[[1, 8]]，允许1到8维，范围过宽。
修复建议:  将weightOptional的dim_num从[[1, 8]]修改为[1]，以准确反映文档中shape为[H]的约束。
是否已修复:  是
------------------------------
错误2：
错误路径:  parameter_constraints[3].constraints.shape[1].constraint[0].dim_num
错误信息:  weightOptional的shape维度数量约束与文档不一致（Atlas A2平台）。文档明确说明weightOptional的shape为[H]，即维度数量固定为1，但JSON中dim_num为[[1, 8]]，允许1到8维，范围过宽。
修复建议:  将weightOptional的dim_num从[[1, 8]]修改为[1]，以准确反映文档中shape为[H]的约束。
是否已修复:  是
------------------------------
错误3：
错误路径:  parameter_constraints[4].constraints.shape[0].constraint[0].dim_num
错误信息:  biasOptional的shape维度数量约束与文档不一致。文档明确说明biasOptional的shape为[H]，即维度数量固定为1，但JSON中dim_num为[[1, 8]]，允许1到8维，范围过宽。
修复建议:  将biasOptional的dim_num从[[1, 8]]修改为[1]，以准确反映文档中shape为[H]的约束。
是否已修复:  是
------------------------------
错误4：
错误路径:  parameter_constraints[4].constraints.shape[1].constraint[0].dim_num
错误信息:  biasOptional的shape维度数量约束与文档不一致（Atlas A2平台）。文档明确说明biasOptional的shape为[H]，即维度数量固定为1，但JSON中dim_num为[[1, 8]]，允许1到8维，范围过宽。
修复建议:  将biasOptional的dim_num从[[1, 8]]修改为[1]，以准确反映文档中shape为[H]的约束。
是否已修复:  是
------------------------------
错误5：
错误路径:  parameter_constraints[1].constraints.shape[0].constraint[0].dim_num
错误信息:  scale的shape维度数量约束与文档不一致。文档说明scale的shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量范围应为1到8。JSON中dim_num为[1]，仅允许1维，范围过窄。
修复建议:  将scale（A3平台）的dim_num从[1]修改为[[1, 8]]，以准确反映文档中shape为[B, H]或[B, 1, H]、B支持0到6维的约束。
是否已修复:  否
------------------------------
错误6：
错误路径:  parameter_constraints[1].constraints.shape[1].constraint[0].dim_num
错误信息:  scale的shape维度数量约束与文档不一致（Atlas A2平台）。文档说明scale的shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量范围应为1到8。JSON中dim_num为[1]，仅允许1维，范围过窄。
修复建议:  将scale（Atlas A2平台）的dim_num从[1]修改为[[1, 8]]，以准确反映文档中shape为[B, H]或[B, 1, H]、B支持0到6维的约束。
是否已修复:  否
------------------------------
错误7：
错误路径:  parameter_constraints[2].constraints.shape[0].constraint[0].dim_num
错误信息:  shift的shape维度数量约束与文档不一致。文档说明shift的shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量范围应为1到8。JSON中dim_num为[1]，仅允许1维，范围过窄。
修复建议:  将shift（A3平台）的dim_num从[1]修改为[[1, 8]]，以准确反映文档中shape为[B, H]或[B, 1, H]、B支持0到6维的约束。
是否已修复:  否
------------------------------
错误8：
错误路径:  parameter_constraints[2].constraints.shape[1].constraint[0].dim_num
错误信息:  shift的shape维度数量约束与文档不一致（Atlas A2平台）。文档说明shift的shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量范围应为1到8。JSON中dim_num为[1]，仅允许1维，范围过窄。
修复建议:  将shift（Atlas A2平台）的dim_num从[1]修改为[[1, 8]]，以准确反映文档中shape为[B, H]或[B, 1, H]、B支持0到6维的约束。
是否已修复:  否
