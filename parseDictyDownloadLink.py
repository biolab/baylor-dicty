#Parse DictyExpress samples table to construct a downloadable link

import glob
import pandas as pd
from collections import OrderedDict
import os
import numpy as np
import matplotlib.pyplot as plt


# Returns 1 if removed
def parse_link_parts(patrt1,part2,part6,project,linksFile1,linksFile2):
    if not (('bowPE' in part6) or ('align' in part2) or ('TopHat' in part6)):
        if type == 'counts':
            url1 = 'https://dictyexpress.research.bcm.edu/data/' + part1 + '/' + part2 + '_expression_rc.tab.gz'
            url2 = 'https://dictyexpress.research.bcm.edu/data/' + part1 + '/expression_rc.tab.gz'
        elif type == 'rpkum':
            url1 = 'https://dictyexpress.research.bcm.edu/data/' + part1 + '/' + part2 + '_expression_rpkum.tab.gz'
            url2 = 'https://dictyexpress.research.bcm.edu/data/' + part1 + '/expression_rpkum.tab.gz'
        elif type == 'rpkum_polya':
            url1 = 'https://dictyexpress.research.bcm.edu/data/' + part1 + '/' + part2 + '_expression_rpkum_polya.tab.gz'
            url2 = 'https://dictyexpress.research.bcm.edu/data/' + part1 + '/expression_rpkum_polya.tab.gz'
        linksFile1.write(part2 + part6 +'_'+project+ '\t' + url1 + '\n')
        linksFile2.write(part2 + part6 +'_'+project+ '\t' + url2 + '\n')
        return 0
    else:
        return 1


#Example: https://dictyexpress.research.bcm.edu/data/5cae387f6b13390493cd3b4d/gbfA_r3_04h_2_S127_L002_R1_001_mapped_expression_rc.tab.gz
#Example table entry: 5ca9aa486b1339abd4cd3b4d,,[],[],Description of tgrC1_r2_12h_16_S32_L002_R1_001_mapped.bam gene expression.,,Gene expressions (tgrC1_r2_12hr_16_S32_L002_R1_001_mapped.bam),"[u'expression', u'profiles']",
#type='counts'
#type='rpkum'
type='rpkum_polya'

#project='milestones'
#project='K_Milestone_ETC'
#project='Tgr_dedifferentiation'
project='All_milestone_mRNA_gff'

#IF file first deleat heading lines with column names
input_type='file'
#input_type='str'

linksFile1=open('/home/karin/Documents/timeTrajectories/data/dicty'+type.upper()+'1_'+project+'.txt','w')
linksFile2=open('/home/karin/Documents/timeTrajectories/data/dicty'+type.upper()+'2_'+project+'.txt','w')
removed=0
if input_type=='file':
    lines=open('/home/karin/Downloads/export_2020-01-10_12-36.csv','r').readlines()
    for line in lines:
        if line != '\n':
            fields = line.split(',')
            part1 = fields[0].replace('"', '')
            part2 = fields[4].split(' ')[2].split('.')[0]
            part6 = fields[6].split(' (')[0].replace(' ', '_')
            removed+=parse_link_parts(part1,part2,part6,project,linksFile1,linksFile2)
else:
    part6 = 'Gene_expression'
    lines="etc.py /srv/dictyexpress_data/5cadcbb76b1339075fcd3b4d/amiB_0h_r1_19_S70_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cadcbb76b1339075fcd3b4c/amiB_2h_r1_20_S71_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cadcbb66b1339075fcd3b4b/amiB_4h_r1_21_S72_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cadcbb66b1339075fcd3b4a/amiB_6h_r1_22_S73_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cadcbb66b1339075fcd3b49/amiB_8h_r1_23_S74_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cadcbb66b13390734cd3b37/amiB_12h_r1_24_S75_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cadcbb56b1339075fcd3b48/amiB_16h_r1_25_S76_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cadcbb56b13392e34cd3b50/amiB_20h_r1_26_S77_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cadcbb56b1339075fcd3b47/amiB_24h_r1_27_S78_L003_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021c96b1339629ccd3b5c/amiB_r2_00h_30_S106_L002_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021c96b13394880cd3b3c/amiB_r2_02h_31_S107_L002_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021c96b1339979fcd3b4b/amiB_r2_04h_32_S108_L002_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021cb6b1339977ccd3b37/amiB_r2_06h_34_S109_L002_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021cc6b13394880cd3b3d/amiB_r2_08h_47_S122_L002_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021cd6b1339977ccd3b38/amiB_r2_12h_35_S110_L002_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021cd6b1339977ccd3b39/amiB_r2_16h_36_S111_L002_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021cd6b13394880cd3b3e/amiB_r2_20h_37_S112_L002_R1_001_mapped_expression_rpkum_polya.tab.gz /srv/dictyexpress_data/5cb021cd6b13394880cd3b3f/amiB_r2_24h_38_S113_L002_R1_001_mapped_expression_rpkum_polya.tab.gz --names 'Gene expressions (amiB_r1_00hr_19_S70_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r1_02hr_20_S71_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r1_04hr_21_S72_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r1_06hr_22_S73_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r1_08hr_23_S74_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r1_12hr_24_S75_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r1_16hr_25_S76_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r1_20hr_26_S77_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r1_24hr_27_S78_L003_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_00hr_30_S106_L002_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_02hr_31_S107_L002_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_04hr_32_S108_L002_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_06hr_34_S109_L002_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_08hr_47_S122_L002_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_12hr_35_S110_L002_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_16hr_36_S111_L002_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_20hr_37_S112_L002_R1_001_mapped.bam)' 'Gene expressions (amiB_r2_24hr_38_S113_L002_R1_001_mapped.bam)' --mean"
    links=[]
    names=[]
    lines=lines.split(' ')
    for line in lines:
        if '.tab.gz' in line:
            links.append(line)
        if '.bam' in line:
            names.append(line)
    if len(links) == len(names):
        for i in range(len(links)):
            part1=links[i].split('/')[3]
            part2=names[i].replace('(','').replace(')','').replace('.bam','').replace("'",'')
            removed+=parse_link_parts(part1,part2,part6,project,linksFile1,linksFile2)
    else:
        print('Error names/links parsing')

linksFile1.close()
linksFile2.close()
print('remouved:',removed)

#In unix: Do in separate foldrs and remove empty files before merging
#Get cookies: ./extract_cookies.sh ~/.mozilla/firefox/frw4paco.default-release/cookies.sqlite > cookies.txt
# grep dictyexpress cookies.txt >cookies_dicty.txt
# while read l;do read fileName url <<< $l; wget -O "$fileName".tab.gz  --load-cookies=/home/karin/Documents/cookies_dicty.txt $url; done < /home/karin/Documents/timeTrajectories/data/dictyRPKUM1_All_milestone_mRNA_gff.txt
# while read l;do read fileName url <<< $l; wget -O "$fileName".tab.gz --load-cookies=/home/karin/Documents/cookies_dicty.txt $url; done < /home/karin/Documents/timeTrajectories/data/dictyRPKUM2_All_milestone_mRNA_gff.txt

#Merge files into one
#path = '/home/karin/Documents/timeTrajectories/data/countsRaw/'
path = '/home/karin/Documents/timeTrajectories/data/RPKUM/'
files = [f for f in glob.glob(path + "*.tab", recursive=True)]
merged=pd.DataFrame()
first=True
for file in files:
    df=pd.read_table(file,low_memory=False)
    fileParts=file.split('/')
    title=fileParts[len(fileParts)-1].replace('_mapped.tab','')
    df.columns=['Gene',title]
    if first:
        merged=df
        first=False
    else:
        merged=pd.merge(df, merged, on = 'Gene',how='outer')
#Format into df with genes as rownames
genes=merged.iloc[:,1:]
genes.index=list(merged.loc[:,'Gene'])
measurments=[]
for measurment in list(genes.columns):
    measurments.append(measurment.rstrip('.tab').replace('-','_').replace('.','_'))
genes.columns=measurments
genes.to_csv('/home/karin/Documents/timeTrajectories/data/countsRaw/combined/mergedGenes_counts.tsv', sep='\t')
#genes.to_csv('/home/karin/Documents/timeTrajectories/data/RPKUM/combined/mergedGenes_RPKUM.tsv', sep='\t')

#Make conditions table

#For counts same as in RPKUM as (genes.columns==conditions["Measurment"]).all() is True


#genes=pd.read_csv('/home/karin/Documents/timeTrajectories/data/countsRaw/combined/mergedGenes.tsv', sep='\t',index_col=0)
genes=pd.read_csv('/home/karin/Documents/timeTrajectories/data/RPKUM/combined/mergedGenes.tsv', sep='\t',index_col=0)
measurments=list(genes.columns)

#Extract data from measurments descriptions
strainsDict={
    'FDPOOL':'AX4',
    'WT':'AX4',
    'AX4':'AX4',
    'TAGB':'tagB',
    'T345':'comH',
    'TGRB1_':'tgrB1',
    'TGRB1C1':'tgrB1C1',
    'TGRC1':'tgrC1',
    'GBFA':'gbfA',
    'AMIB':'amiB',
    'CUDA':'cudA',
    'DGCA':'dgcA',
    'ECMA':'ecmA',
    'GTAG':'gtaG',
    'II3':'ii3',
    'PKAC25':'pkaCoeAX4',
    'PKACOE25':'pkaCoeAX4',
    'MYBBGFP':'mybBGFP',
    'MYBB_':'mybB',
    'PKAR':'pkaR',
    'AC3_PKAC':'ac3pkaCoe',
    'GTAI':'gtaI',
    '_GTAC_':'gtaC',
    'ACAA':'acaA',
    }
reps=OrderedDict({
      'rep1':'rep1',
      'rep2':'rep2',
      '_r1_':'r1',
      '_r2_':'r2',
      '_r3_':'r3',
    '_r4_':'r4',
    'Pool26': 'Pool26',
    'FDpool02':'FDpool02',
      'pool27':'pool27',
    'Pool29': 'Pool2829',
    'Pool28': 'Pool2829',

    'bio1': 'bio1',
    'biorep2': 'bio2',
    'bio2': 'bio2',
    'bio3': 'bio3',
    'bio4': 'bio4',
    #Based on timepoints pool28&29 AX4 are the same
      'FDpool01':'FDpool01',
      'pool36':'pool36',
      'pool35':'pool35',
      'pool38':'pool38',
      'pool30':'pool30',
      'pool_21':'pool21',
      'pool_19':'pool19',
      '_3_3_':'33',
      })
additional_reps=OrderedDict({
    'gtaC':['1A','7B']
})
reps_experiment=OrderedDict({
    'TOPHAT':'1',
    'ALIGN':'2',
    'BOWPE':'3'
})
times=[]
strains=[]
include=[]
replicate_ids=[]
had_multiple_ids=[]
for measurment in measurments:
    mUpper=measurment.upper()
    strain=[]
    for strainID,strainName in strainsDict.items():
        if strainID in mUpper:
            strain.append(strainName)
    if len(strain)==1:
        strains.append(strain[0])
    else:
        strains.append('NaN')
        print('Strain', measurment)
    mParts=mUpper.split('_')
    time=[]
    for part in mParts:
        if 'H' in part:
            if '-' in part:
                part=part.split('-')[0]
            partPruned=part.replace('H','').replace('R','')
            if partPruned.isdigit():
                time.append(partPruned)
    if len(time)==1:
        times.append(int(time[0]))
    else:
        times.append('NaN')
        print('Time',measurment)
    description = measurment
    present_rep = []
    for key in reps.keys():
        if key in description:
            present_rep.append(key)
    present_rep_experiment = []
    for key in reps_experiment.keys():
        if key in mUpper:
            present_rep_experiment.append(reps_experiment[key])
    #Adds preprocessing type to replicate
    #rep_experiment=''
    #for experiment in present_rep_experiment:
        #rep_experiment+=experiment
    if len(present_rep_experiment)>0:
        print('Wrong processing: '+measurment)
    if len(present_rep) > 0:
        value = reps[present_rep[0]]
        if strain[0] in additional_reps.keys():
            for additional_id in additional_reps[strain[0]]:
                if additional_id in measurment:
                    value+=additional_id

        #replicate_ids.append(strain[0] + '_' + value+rep_experiment)
        replicate_ids.append(strain[0] + '_' + value )
    if len(present_rep) > 1:
        had_multiple_ids.append(measurment)
    elif len(present_rep) < 1:
        print('Replicate',measurment)
   # if ('pool_21' in measuremnt) or ('pool_19' in measuremnt):
    #    include.append('FALSE')
   # else:
     #   include.append('TRUE')
#Make conditions df
zippedList =  list(zip(measurments, strains, times,replicate_ids
                      # ,include
                        ))
# Create a dataframe from zipped list
conditions = pd.DataFrame(zippedList, columns=['Measurment', 'Strain', 'Time','Replicate'
                                                #,'Include'
                                               ])

#conditions.to_csv('/home/karin/Documents/timeTrajectories/data/countsRaw/combined/conditions_mergedGenes.tsv', sep='\t',index=False)
conditions.to_csv('/home/karin/Documents/timeTrajectories/data/RPKUM/combined/conditions_mergedGenes.tsv', sep='\t',index=False)

#Create table of replicates and time points:
time_points=list(set(conditions.loc[:,"Time"]))
time_points.sort()
replicates=list(set(conditions.loc[:,"Replicate"]))
replicates.sort()
time_matrix=pd.DataFrame(np.zeros((len(replicates),len(time_points))))
time_matrix.index=replicates
time_matrix.columns=time_points
for row in conditions.iterrows():
    row=row[1]
    time=row["Time"]
    rep=row["Replicate"]
    time_matrix.loc[rep,time] += 1

time_matrix.to_csv('/home/karin/Documents/timeTrajectories/data/RPKUM/combined/time_points.tsv',sep='\t')



#Check expression distribution:
genes.boxplot(flierprops=dict(markersize=1))
plt.yscale('log')

#Check for AX4_Pool261 & AX4_Pool26 and AX4_bio11 & AX4_bio1 and pkaR_bio1 & pkaR_bio2 and mybB_bio13 &mybB_bio1
genes.T[list((conditions["Replicate"]=='AX4_Pool261') | (conditions["Replicate"]=='AX4_Pool26'))].T.boxplot()
#RESULT: Top hat gene counts are higher (median based) (eg. pool261 and bio11); same for pkaR_bio1 compared to bio2 and mybB_bio13 compared to mybB_bio1


#OLD - when replicates were added latter
conditions=pd.read_csv('/home/karin/Documents/timeTrajectories/data/countsRaw/combined/conditions.tsv', sep='\t')
reps=OrderedDict({'FDpool02':'A',
      'pool27':'B',
      'Pool26':'C',
      #Based on timepoints pool28&29 are the same
      'Pool29':'D',
      'Pool28': 'D',
      'FDpool01':'E',
      'rep1':'F',
      'rep2':'G',
      '_r1_':'I',
      '_r2_':'J',
      '_r3_':'K',
      'pool36':'L',
      'pool35':'H',
      'pool38':'P',
      'pool30':'R',
      'pool_21':'N',
      'pool_19':'O',
      'bio2': 'M',
      })
replicate_ids=[]
had_multiple=[]
for index, row in conditions.iterrows():
    present=[]
    description=row['Measurment']
    for key in reps.keys():
        if key in description:
            present.append(key)
    value=reps[present[0]]
    replicate_ids.append(row['Strain']+'_'+value)
    if len(present) is not 1:
        had_multiple.append(row['Measurment'])

conditions.insert(1,'Replicate',replicate_ids)
include=conditions['Include']
include=include.replace(True,'TRUE')
include=include.replace(False,'FALSE')
del conditions['Include']
conditions.insert(3,'Include',include.to_list())
conditions.to_csv('/home/karin/Documents/timeTrajectories/data/countsRaw/combined/conditions.tsv', sep='\t',index=False)

#Find missing files:
files=[]
for r, d, f in os.walk('/home/karin/Documents/timeTrajectories/data/RPKUM/milestones/'):
    if len(f)>1:
        files=f
files2=[]
for file in files:
    files2.append(file.rstrip('.tab.gz'))
lines=open('/home/karin/Downloads/export_2019-10-22_17-31.csv','r').readlines()
missing=[]
encountered=[]
for line in lines:
    if line!='\n':
        fields=line.split(',')
        part2=fields[4].split(' ')[2].split('.')[0]
        if part2 in encountered:
            print(part2)
        encountered.append(part2)
        if part2 not in files2:
            missing.append(part2)
