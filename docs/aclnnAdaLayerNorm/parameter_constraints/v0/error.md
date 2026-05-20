错误1：
错误路径:  parameter_constraints[3].constraints.shape[0].constraint[0].dim_num
错误信息:  weightOptional的shape维度数量约束与文档不一致。文档明确说明weightOptional的shape为[H]，即维度数量固定为1，但JSON中dim_num为[[1, 8]]，允许1到8维，范围过宽。
修复建议:  将weightOptional的dim_num从[[1, 8]]修改为[1]，以准确反映文档中shape为[H]的约束。
是否已修复:  否
------------------------------
错误2：
错误路径:  parameter_constraints[3].constraints.shape[1].constraint[0].dim_num
错误信息:  weightOptional的shape维度数量约束与文档不一致（Atlas A2平台）。文档明确说明weightOptional的shape为[H]，即维度数量固定为1，但JSON中dim_num为[[1, 8]]，允许1到8维，范围过宽。
修复建议:  将weightOptional的dim_num从[[1, 8]]修改为[1]，以准确反映文档中shape为[H]的约束。
是否已修复:  否
------------------------------
错误3：
错误路径:  parameter_constraints[4].constraints.shape[0].constraint[0].dim_num
错误信息:  biasOptional的shape维度数量约束与文档不一致。文档明确说明biasOptional的shape为[H]，即维度数量固定为1，但JSON中dim_num为[[1, 8]]，允许1到8维，范围过宽。
修复建议:  将biasOptional的dim_num从[[1, 8]]修改为[1]，以准确反映文档中shape为[H]的约束。
是否已修复:  否
------------------------------
错误4：
错误路径:  parameter_constraints[4].constraints.shape[1].constraint[0].dim_num
错误信息:  biasOptional的shape维度数量约束与文档不一致（Atlas A2平台）。文档明确说明biasOptional的shape为[H]，即维度数量固定为1，但JSON中dim_num为[[1, 8]]，允许1到8维，范围过宽。
修复建议:  将biasOptional的dim_num从[[1, 8]]修改为[1]，以准确反映文档中shape为[H]的约束。
是否已修复:  否
------------------------------
错误5：
错误路径:  parameter_constraints[1].constraints.shape[0].constraint[0].dim_num
错误信息:  scale的shape维度数量约束与文档不一致。文档说明scale的shape为[B, H]或[B, 1, H]，其中B支持0到6维。当B为0维时，shape为[H]（1维）或[1, H]（2维），最低维度数应为1；但结合B最多6维的情况，[B, H]最多7维，[B, 1, H]最多8维。JSON中dim_num为[[1, 8]]虽然包含了1维的情况，但文档示例shape为[B, H]或[B, 1, H]，实际最低应为2维（当B至少为1时），存在不一致。
修复建议:  根据文档中scale的shape描述[B, H]或[B, 1, H]，确认B的维度范围后精确设定dim_num。如果B为0到6维，则dim_num应为[[1, 8]]；如果B至少为1维，则应为[[2, 8]]。需与算子实际行为对齐。
是否已修复:  否
------------------------------
错误6：
错误路径:  parameter_constraints[1].constraints.shape[1].constraint[0].dim_num
错误信息:  scale的shape维度数量约束与文档不一致（Atlas A2平台）。同上，需确认B的维度范围以精确设定dim_num。
修复建议:  同上，根据文档确认B的维度范围后精确设定dim_num。
是否已修复:  否
------------------------------
错误7：
错误路径:  parameter_constraints[2].constraints.shape[0].constraint[0].dim_num
错误信息:  shift的shape维度数量约束与文档不一致。文档说明shift的shape为[B, H]或[B, 1, H]，与scale相同，JSON中dim_num为[[1, 8]]，存在与scale相同的一致性问题。
修复建议:  根据文档中shift的shape描述[B, H]或[B, 1, H]，确认B的维度范围后精确设定dim_num。
是否已修复:  否
------------------------------
错误8：
错误路径:  parameter_constraints[2].constraints.shape[1].constraint[0].dim_num
错误信息:  shift的shape维度数量约束与文档不一致（Atlas A2平台）。同上，需确认B的维度范围以精确设定dim_num。
修复建议:  同上，根据文档确认B的维度范围后精确设定dim_num。
是否已修复:  否
