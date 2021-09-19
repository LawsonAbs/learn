'''
Author: LawsonAbs
Date: 2021-01-13 08:40:25
LastEditTime: 2021-01-13 08:51:36
FilePath: /argparse/example.py

给出 argparse 使用的案例
'''

import argparse

# step1.创建一个 ArgumentParser 对象，该对象包涵将命令行解析成python数据类型所需的全部信息
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))