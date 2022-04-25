#!/usr/bin/env python
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."
import numpy as np  # notes are in main_test
import os, csv
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--csv_GMMin", default='/home/io18230/Desktop/C_GMM_SELECT_DIS.csv', type=str)
parser.add_argument("--csv_GMMout", default='/home/io18230/Desktop/C_GMM_SELECT_DInew.csv', type=str)
#parser.add_argument("--csv_GMM", default='/home/io18230/Desktop/output/1-200_1-200/3visual/current035_retrain_for_animal/C_GMM_SELECT_DIS.csv', type=str)
parser.add_argument("--img_path", default='/home/io18230/Desktop/RGB (copy)', type=str)
parser.add_argument("--out_path", default='/home/io18230/Desktop/', type=str)
parser.add_argument("--merge", default=0, type=int)
args = parser.parse_args()

show_negtive = 0

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def connection_check(list_all_pair):
    again = 0
    print(f'connection_check: ',end='')
    list_m = []
    for i, item in enumerate(list_all_pair):
        for it in item:
            for j, refe in enumerate(list_all_pair):
                if it in refe and i != j:
                    list_m.append([i, j])
                    again = 1
                    print('Merged', i, 'and', j, ',set', j, 'to [].' ,end='')
                    if len (list_all_pair[j])>0:
                        tmp = list(set(list_all_pair[j]+list_all_pair[i]))
                        tmp.sort()
                        list_all_pair[i] = tmp
                        list_all_pair[j] = []
    return list_all_pair, again

def gmm_filter_multiple(args,annotation=0):
    csvFile = csv.reader(open(args.csv_GMMout, 'r'), delimiter=',')
    reader = list(csvFile)
    del reader[0]
    gmm_distance = []
    gmm_forder1 = []
    gmm_forder2 = []
    gmm_l1 = []
    gmm_l2 = []
    gmm_pri = []

    for i in range(len(reader)):
        if reader[i][1] != '':
            gmm_distance.append(reader[i][1])
            gmm_forder1.append(reader[i][2])
            gmm_forder2.append(reader[i][3])
            gmm_l1.append(reader[i][4])
            gmm_l2.append(reader[i][5])
            gmm_pri.append(reader[i][6])

    #filter repeated
    print(f'Original question length :', len(gmm_l2)) # yes (including over 314)

    # calculate the number of questions
    gmm_count_raw = 0
    gmm_count_raw_yes = 0
    f1new = [] #folder
    f2new = []
    l1new = [] # label
    l2new = []
    dnew = []
    pnew = []

    # skip 重复的
    for i,(l1,l2,f1,f2) in enumerate(zip(gmm_l1,gmm_l2,gmm_forder1,gmm_forder2)):
        if f1 == f2:   # filter self  like 020/  020/1
            continue
        else:          # 删除互文
            skip_mark = 0
            if f1 in f1new:
                index = [i for i,val in enumerate(f1new) if val==f1]
                for item in index:
                    if f2 == f2new[item]:
                        skip_mark = 1
                        #print(f'skip it')
            if f1 in f2new:
                index = [i for i,val in enumerate(f2new) if val==f1]
                for item in index:
                    if f2 == f1new[item]:
                        skip_mark = 1
                        #print(f'skip it')
            if not skip_mark:
                if int(f1[:3]) > int(f2[:3]):
                    f1new.append(f2)
                    f2new.append(f1)
                    l1new.append(l2)
                    l2new.append(l1)
                else:
                    f1new.append(f1)
                    f2new.append(f2)
                    l1new.append(l1)
                    l2new.append(l2)
                dnew.append(gmm_distance[i])
                pnew.append(gmm_pri[i])

    # count number
    aaa_distance = []
    aaa_forder1 = []
    aaa_forder2 = []
    aaa_forder1n=[]
    aaa_forder2n=[]
    aaa_l1 = []
    aaa_pri= []
    for i, (l1, l2) in enumerate(zip(l1new, l2new)):  # the pair filtered self and repeted things
        if l1 == l2:
            gmm_count_raw_yes += 1
            aaa_distance.append(dnew[i])
            aaa_forder1.append(f1new[i])
            aaa_forder2.append(f2new[i])
            aaa_l1.append(l1)
            aaa_pri.append(gmm_pri[i])
        else:
            if int(f1new[i][:3]) > 400 or int(f2new[i][:3]) > 400:   #314
                if show_negtive:
                    print(f1new[i], f2new[i], 'are not the same cattle, over 314')
            else:
                if show_negtive:
                    print(f1new[i], f2new[i], 'are not the same cattle')
            aaa_forder1n.append(f1new[i])
            aaa_forder2n.append(f2new[i])
        gmm_count_raw += 1
    # data2 = pd.DataFrame(
    #     {'Distance': dnew, 'folder1': f1new, 'folder2': f2new, 'label1': l1new, 'label2': l2new, 'priority': pnew})
    # data2.to_csv(os.path.join(args.out_path, 'C_gmm_distance_filter.csv'))


    print(f'\nReal Q (delete repeat):', gmm_count_raw, 'original questions')
    print(f'True matches pairs    :' , gmm_count_raw_yes, f'  (larger than should merge)') #postive questions
    print('Rate                  :',round(gmm_count_raw_yes*100 / gmm_count_raw,2),'%')
    #print('Number of Negtive q   :', gmm_count_raw-gmm_count_raw_yes)


    if annotation:
        list_all_pair = []

        for item in sorted(os.listdir(args.img_path)):
            for subitm in sorted(os.listdir(os.path.join(args.img_path, item))):
                pathName = item + '/' + subitm
                if pathName in aaa_forder1 and os.path.exists(os.path.join(args.img_path, pathName)):
                    # pathName this lead to better match training folders
                    found_posi_flag = 0
                    for index in range(len(aaa_forder1)):
                        if pathName == aaa_forder1[index]:
                            found_posi_flag = 1
                            #print( f'mearging' ,pathName, 'and', aaa_forder2[index])
                            sec_pathName = aaa_forder2[index]
                            # attach every new pair
                            list_all_pair.append([pathName, sec_pathName])
                            count = 0

                            for idex, each in enumerate(list_all_pair[:-1]):
                                if (pathName in each and sec_pathName not in each) or (pathName not in each and sec_pathName in each):
                                    count += 1
                                    if pathName in each:
                                        list_all_pair[idex] = each + [sec_pathName]
                                    else:
                                        list_all_pair[idex] = each + [pathName]

                                    #print(idex, pathName, sec_pathName, each)
                                if (pathName in each and sec_pathName in each):
                                    count += 1
                                    #print(idex,pathName,sec_pathName,each)
                            if count >= 1:
                                list_all_pair.pop()
                    if not found_posi_flag:
                        m = 1



    return list_all_pair, gmm_count_raw - gmm_count_raw_yes, [aaa_forder1n,aaa_forder2n]


def delete_repeat(args):
    csvFile = csv.reader(open(args.csv_GMMin, 'r'), delimiter=',')
    reader = list(csvFile)
    del reader[0]
    gmm_distance = []
    gmm_forder1 = []
    gmm_forder2 = []
    gmm_l1 = []
    gmm_l2 = []
    gmm_pri = []

    for i in range(len(reader)):
        if reader[i][1] != '':
            gmm_distance.append(reader[i][1])
            gmm_forder1.append(reader[i][2])
            gmm_forder2.append(reader[i][3])
            gmm_l1.append(reader[i][4])
            gmm_l2.append(reader[i][5])
            gmm_pri.append(reader[i][6])

    #filter repeated
    print(f'Original question length:', len(gmm_l2))

    # calculate the number of questions
    gmm_count_raw = 0
    f1new = []
    f2new = []
    l1new = []
    l2new = []

    list_remove = []
    # skip 重复的
    for i,(l1,l2,f1,f2) in enumerate(zip(gmm_l1,gmm_l2,gmm_forder1,gmm_forder2)):
        if f1 == f2:   # filter self  like 020/  020/1
            list_remove.append(i)
            continue
        elif f1[:3] == f2[:3]:  # the same folder
            list_remove.append(i)
            print('remove same folder')
            continue
        else:          # 删除互文
            skip_mark = 0
            if f1 in f1new:
                index = [i for i,val in enumerate(f1new) if val==f1]
                for item in index:
                    if f2 == f2new[item]:
                        skip_mark = 1
                        list_remove.append(i)
                        #print(f'skip it')
            if f1 in f2new:
                index = [i for i,val in enumerate(f2new) if val==f1]
                for item in index:
                    if f2 == f1new[item]:
                        skip_mark = 1
                        list_remove.append(i)
                        #print(f'skip it')
            if not skip_mark:
                if int(f1[:3]) > int(f2[:3]):
                    f1new.append(f2)
                    f2new.append(f1)
                    l1new.append(l2)
                    l2new.append(l1)
                else:
                    f1new.append(f1)
                    f2new.append(f2)
                    l1new.append(l1)
                    l2new.append(l2)


    # count number
    for i, (l1, l2) in enumerate(zip(l1new, l2new)):  # the pair filtered self and repeted things
        gmm_count_raw += 1

    print(f'Real Q (delete repeat):', gmm_count_raw)
    data = pd.read_csv(args.csv_GMMin)
    # print(list_remove)
    data_new = data.drop(list_remove)
    data_new.to_csv(args.csv_GMMin[:-5]+"new.csv",index=0)

if __name__ == '__main__':

    delete_repeat(args)

    list_all_pair, negtive_q, list_n = gmm_filter_multiple(args, annotation=1)
    again = 1
    while again:
        list_all_pair, again = connection_check(list_all_pair)

    # calculated the number should be merged
    final = []
    count = 0
    for i in list_all_pair:
        if len(i)>0:
            final.append(i)
            for item in i:
                count += 1
                if int(item[:3])>400: #314
                    count -= 1
                    # keep them in the output already.
                    #print('do not count over 314')
        if len(i) > 4:
            # for test check
            pass

    print('\n')
    print('Total Questions    :', negtive_q + count - len(final))
    print('Should merge       :', count-len(final), '    tracklets')
    print('useful rate        :', round((count-len(final))/(negtive_q+count-len(final))*100,2), '%')
    print('Number of Negtive q:', negtive_q) # I know there are repeated questions



    listcandidate = []
    listanchor = []
    for item in final:
        listanchor.append(item[0])
        listcandidate.append(item[1:])
    data1 = pd.DataFrame(
        {'listLabel': listanchor, 'listanchor': listanchor, 'listcandidate': listcandidate, 'priority': listanchor,
         'Dist_average': listanchor})
    print(f'\nsave to', os.path.join(args.out_path, 'C_annotator.csv'))
    data1.to_csv(os.path.join(args.out_path, 'C_annotator.csv'))