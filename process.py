# import xlrd
# #打开xls文件
# data1 = xlrd.open_workbook(r'C:\Users\雁来月暮秋\Desktop\\flickr_attribute.xlsx')
# #从文件中通过表名找到表
# table = data1.sheet_by_name('Sheet1')
# f_w=open('data/flickr.attributes','w')
#
# i=0
# f_w.write("#Nodes	3312\n#attributes	3703\n")
# for rown in range(table.nrows):
#     f_w.write(str(i)+"\t")
#     for value in table.row_values(rown):
#         f_w.write(str(value)+"\t")
#     i=i+1
#     f_w.write("\n")
#

# import xlrd
# #打开xls文件
# data1 = xlrd.open_workbook(r'C:\Users\雁来月暮秋\Desktop\\flickr_attribute.xlsx')
# #从文件中通过表名找到表
# table = data1.sheet_by_name('Sheet1')
# f_w=open('data/flickr.attributes','w')
#
# i=0
# f_w.write("#Nodes	3312\n#attributes	3703\n")
# for rown in range(table.nrows):
#     f_w.write(str(i)+"\t")
#     for value in table.row_values(rown):
#         f_w.write(str(value)+"\t")
#     i=i+1
#     f_w.write("\n")

f_r=open("data/Flickr.node",'r')
attribute_matrix=[[0]*12047 for i in range(7575)]
attribute_line=f_r.readlines()
attribute_line.pop(0)
attribute_line.pop(0)
for line in attribute_line:
    line=line.strip("\n").split("\t")
    node_id=int(line[0])
    attribute_id=int(line[1])
    attribute_matrix[node_id][attribute_id]=1
print(attribute_matrix[800])
f_w=open('data/Flickr.attributes','w')
f_w.write("#Nodes	7575\n#Attributes	12047\n")
for i in range(7575):
    f_w.write(str(i)+'\t')
    for j in range(12047):
        f_w.write(str(attribute_matrix[i][j])+"\t")
    f_w.write("\n")