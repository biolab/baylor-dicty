a<-c(1, 1, 1, 2, 3, 3, 4, 1)
b<-c(1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 7, 8)
ks.test(a,b,alternative='two.sided') # D = 0.34559, p-value = 0.5343
ks.test(a,b,alternative='less') #D^- = -1.3878e-17, p-value = 1
ks.test(a,b,alternative='greater') #D^+ = 0.34559, p-value = 0.2727
ks.test(b,a,alternative='two.sided') #D = 0.34559, p-value = 0.5343
ks.test(b,a,alternative='less') #D^- = 0.34559, p-value = 0.2727
ks.test(b,a,alternative='greater') #D^+ = -1.3878e-17, p-value = 1
#p_two/2=similar but not equal to one tailed in right direction
#However with bigger samples the p values are the same (eg. a=rep(a,times=100)) 