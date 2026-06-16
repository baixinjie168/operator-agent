#!/usr/bin/env python3
"""清空所有算子文档解析数据"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'operator_agent.db')

# document_versions 的子表
CHILD_TABLES = [
    'constraints_result',
    'dtype_combinations',
    'function_signatures',
    'param_relations',
    'parameters',
    'platform_support',
    'return_codes',
    'task_items',
]

# 主表
MAIN_TABLES = [
    'document_versions',
    'operators',
    'tasks',
]

ALL_TABLES = CHILD_TABLES + MAIN_TABLES


def clean_all():
    print(f'Database: {DB_PATH}')
    if not os.path.exists(DB_PATH):
        print('Database file not found!')
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA foreign_keys = ON')

    # 统计删除前的数据量
    print()
    print('--- 删除前 ---')
    for table in ALL_TABLES:
        count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        print(f'  {table}: {count} rows')

    # 执行删除
    cursor = conn.cursor()
    cursor.execute('BEGIN TRANSACTION')

    for table in CHILD_TABLES:
        cursor.execute(f'DELETE FROM {table}')

    for table in MAIN_TABLES:
        cursor.execute(f'DELETE FROM {table}')

    # 重置自增序列
    placeholders = ','.join(['?'] * len(ALL_TABLES))
    cursor.execute(f'DELETE FROM sqlite_sequence WHERE name IN ({placeholders})', ALL_TABLES)

    conn.commit()

    # 统计删除后
    print()
    print('--- 删除后 ---')
    for table in ALL_TABLES:
        count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        print(f'  {table}: {count} rows')

    conn.close()
    print()
    print('Done. All data cleaned.')


if __name__ == '__main__':
    confirm = input('This will delete ALL parsed data. Continue? (y/N): ')
    if confirm.lower() == 'y':
        clean_all()
    else:
        print('Cancelled.')
