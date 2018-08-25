import os

fil_phoList='pho_list.txt'
phoList=open(fil_phoList,'r')

dir_observed='observed/'
dir_pho='pho/'
unfound_pho=0
print('Moving PHOs to their own folder...')
for pho_num in phoList:
    pho_num=pho_num.strip('\n')
    fil_oo='object_'+pho_num+'.npy'
    try:
        os.rename(dir_observed+fil_oo, dir_pho+fil_oo)
    except OSError:
        unfound_pho+=1
print('Done!')
print('%s PHOs were not found'%(unfound_pho))
