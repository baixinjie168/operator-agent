-- ============================================================
-- 清空所有算子文档解析数据
-- 使用: sqlite3 data/operator_agent.db < scripts/clean_all_data.sql
--   或: python scripts/clean_all_data.py
-- ============================================================

PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- 1. 删除 document_versions 的所有子表数据
DELETE FROM constraints_result;
DELETE FROM dtype_combinations;
DELETE FROM function_signatures;
DELETE FROM param_relations;
DELETE FROM parameters;
DELETE FROM platform_support;
DELETE FROM return_codes;

-- 2. 删除 document_versions
DELETE FROM document_versions;

-- 3. 删除 operators
DELETE FROM operators;

-- 4. 删除 task_items 和 tasks
DELETE FROM task_items;
DELETE FROM tasks;

-- 5. 重置自增序列
DELETE FROM sqlite_sequence WHERE name IN (
    'operators', 'document_versions', 'constraints_result',
    'dtype_combinations', 'function_signatures', 'param_relations',
    'parameters', 'platform_support', 'return_codes',
    'task_items', 'tasks'
);

COMMIT;
