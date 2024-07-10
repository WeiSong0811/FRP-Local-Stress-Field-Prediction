import os
import subprocess
from odbAccess import *
from abaqusConstants import *
import numpy as np

def check_and_upgrade_odb(odb_file):
    try:
        odb = openOdb(path=odb_file)
        odb.close()
        return odb_file  # 返回现有文件名
    except Exception as e:
        if 'OdbError' in str(e):
            print(f"需要升级的 .odb 文件: {odb_file}")
            new_odb_file = 'upgraded_' + odb_file
            upgrade_command = f'abaqus upgrade -job {new_odb_file} -odb {odb_file}'
            print(f"执行命令: {upgrade_command}")
            result = subprocess.run(upgrade_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to upgrade odb file: {result.stderr.decode()}")
            print(f"升级后的 .odb 文件: {new_odb_file}")
            return new_odb_file  # 返回升级后的文件名
        else:
            raise

def extract_and_save_data(odb_file, output_file):
    print(f"打开 .odb 文件: {odb_file}")
    odb = openOdb(path=odb_file)

    # 选择步和帧
    step = odb.steps['Loading_step']  # 替换为找到的步名称
    frame = step.frames[-1]     # 最后一帧
    print("步和帧已选择")

    # 选择实例和所有节点
    instance_name = 'PVE56-1'  # 替换为找到的实例名称
    instance = odb.rootAssembly.instances[instance_name]
    all_nodes = instance.nodes
    print(f"实例名称: {instance_name}, 节点总数: {len(all_nodes)}")

    # 提取应力数据
    stress = frame.fieldOutputs['S']
    stressField = stress.getSubset(region=instance)
    print("应力数据已提取")

    # 提取应变数据
    strain = frame.fieldOutputs['E']
    strainField = strain.getSubset(region=instance)
    print("应变数据已提取")

    # 提取所有节点的数据
    nodeCoords = [node.coordinates for node in all_nodes]
    nodeStresses = [stressField.values[i].data for i in range(len(all_nodes))]
    nodeStrains = [strainField.values[i].data for i in range(len(all_nodes))]
    print("所有节点坐标、应力和应变数据已获取")

    # 将数据转换为NumPy数组
    nodeCoords = np.array(nodeCoords)
    nodeStresses = np.array(nodeStresses)
    nodeStrains = np.array(nodeStrains)
    print("数据已转换为NumPy数组")
    print(f"节点坐标数组形状: {nodeCoords.shape}")
    print(f"节点应力数组形状: {nodeStresses.shape}")
    print(f"节点应变数组形状: {nodeStrains.shape}")

    # 导出数据到txt文件
    with open(output_file, 'w') as f:
        f.write("X_Coord,Y_Coord,Z_Coord,S11,S22,S33,S12,S13,S23,E11,E22,E33,E12,E13,E23\n")
        for coord, stress, strain in zip(nodeCoords, nodeStresses, nodeStrains):
            f.write(f"{coord[0]},{coord[1]},{coord[2]},{stress[0]},{stress[1]},{stress[2]},{stress[3]},{stress[4]},{stress[5]},{strain[0]},{strain[1]},{strain[2]},{strain[3]},{strain[4]},{strain[5]}\n")
    print(f"数据已导出到 {output_file} 文件")

    # 验证文件内容
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print("文件内容预览：")
        for line in lines[:5]:  # 只打印前5行进行验证
            print(line.strip())

    # 关闭 .odb 文件
    odb.close()
    print(".odb 文件已关闭")

# 主程序
if __name__ == "__main__":
    odb_file = 'C:/Users/weiso/Desktop/DA/test0807/0.odb'  # 替换为你的 .odb 文件名
    output_file = 'C:/Users/weiso/Desktop/DA/test0807/node_data.txt'  # 替换为你希望保存的路径和文件名
    try:
        upgraded_odb_file = check_and_upgrade_odb(odb_file)
        extract_and_save_data(upgraded_odb_file, output_file)
    except Exception as e:
        print(f"An error occurred: {e}")
