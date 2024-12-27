import pandas as pd

def generate_table_full_name_dict(excel_path='data/数据字典/数据字典.xlsx'):
    """
    从数据字典Excel中生成 {'表英文': '库名英文.表英文'} 的字典
    
    Args:
        excel_path (str): 数据字典Excel文件路径
    
    Returns:
        dict: 表全名字典
    """
    try:
        # 读取库表关系sheet
        df = pd.read_excel(excel_path, sheet_name='库表关系')
        
        # 创建字典
        table_full_name_dict = {}
        
        # 遍历DataFrame
        for _, row in df.iterrows():
            database_en = row.get('库名英文', '')
            table_en = row.get('表英文', '')
            
            # 确保库名和表名都不为空
            if database_en and table_en:
                full_table_name = f"{database_en}.{table_en}"
                table_full_name_dict[table_en] = full_table_name
        
        return table_full_name_dict
    
    except Exception as e:
        print(f"生成表全名字典时发生错误: {str(e)}")
        return {}

# 测试函数
def test_table_full_name_dict():
    table_full_name_dict = generate_table_full_name_dict()
    
    print("表全名字典:")
    for table_en, full_name in table_full_name_dict.items():
        print(f"{table_en}: {full_name}")
    
    print(f"\n总表数: {len(table_full_name_dict)}")

if __name__ == "__main__":
    test_table_full_name_dict()