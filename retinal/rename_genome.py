genome_path='/home/karin/Documents/retinal/data/NCBI_genome_GB_GCA_000364345.1_Macaca_fascicularis_5.0_genomic.fna'
genome=open(genome_path,'r').readlines()
for line,idx in  zip(genome,range(len(genome))):
    if '>' in line:
        chromosome=None
        main_chr=''
        if 'chromosome' in line:
            chromosome=line.split('chromosome ')[1].split(' ')[0].replace(',','')
        else:
            chromosome='Un'
        if 'unlocalized'  in line:
            main_chr='_random'
        first='_'+line.split(' ')[0].replace('>','').split('.')[0]
        if chromosome !='Un' and main_chr=='':
            first=''
        new_line='>chr'+chromosome+first+main_chr
        genome[idx]=new_line+'\n'

genome_parsed=open('/home/karin/Documents/retinal/data/parsed_NCBI_genome_GB_GCA_000364345.1_Macaca_fascicularis_5.0_genomic.fna','w')
genome_parsed.writelines(genome)