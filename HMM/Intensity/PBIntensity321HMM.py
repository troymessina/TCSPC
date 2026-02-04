# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:48:34 2021

@author: troy c. messina
"""
import math
import pandas as pd
from scipy import stats
from scipy.stats import poisson
from scipy.stats import norm
from scipy.special import factorial
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
cp = []

##################Segment_Recurse###################
## Find segments that have different count rates ##
####################################################
def Segment_Recurse(data, mark1, mark2):
    global cp
    stop = 0
    minpts=4
    StudT = np.zeros(np.size(data))
    while stop==0:
        for jj in range(mark1+1, mark2-1):
            #move a marker through the trajectory
            #get stats to the left
            aveL = np.mean(data[mark1:jj])
            sigL = max(np.std(data[mark1:jj]), np.std(data)/4)#sometimes there is only one point
            nptsL = max(np.fabs(jj-mark1),1)#make sure at least 1 to not divide by 0
            #get stats to right of marker
            aveR = np.mean(data[jj+1:mark2])
            sigR = max(np.std(data[jj+1:mark2]),np.std(data)/4)
            nptsR = max(np.fabs(mark2 - jj+1),1)#make sure at least 1 to not divide by 0
            if aveL==0 or sigL==0 or nptsL==0 or aveR==0 or sigR==0 or nptsR==0:
                print(aveL, sigL, nptsL, aveR, sigR, nptsR)
            #calculate student-T of left and right segments with weigthed sigma
            tval = np.fabs(aveL - aveR)/((sigL*nptsL+sigR*nptsR)/(nptsL+nptsR) * np.sqrt(1/nptsL+1/(nptsR)))#calculate student-T of left and right segments with weighted sigma
            if float('-inf') < float(tval) < float('inf'):
                StudT[jj] = tval
            else:
                StudT[jj] = 0
		#endfor
        maxT = np.nanmax(StudT[mark1:mark2])
        prev_mark = mark1
        mark1 = np.nanargmax(StudT[prev_mark:mark2])+prev_mark
        #print(prev_mark, mark1, mark2)
        #print(maxT)#, mark1, np.absolute(mark2-prev_mark))
        #require at least 3 data points and t-value high
        if np.fabs(mark1-prev_mark)>minpts and np.fabs(mark2-prev_mark)>minpts and maxT > stats.t.ppf(0.99, np.absolute(mark2-prev_mark), loc=aveR):
            cp = np.append(cp, mark1)
            x = np.where(cp==mark1)
            if np.size(x)>1:
                cp = np.unique(cp)
            Segment_Recurse(data, prev_mark, mark1) #recurse on the left side
            Segment_Recurse(data, mark1, mark2)#recurse on the right side
        else:
            stop=1
        stop=1
   # print(maxT, mark1, cp)
    return StudT, cp
######################End Segment_Recurse#########################

######################## Make_Rates ###############################
## take the list of change points and make a list of count rates ##
## and transition rates ###########################################
###################################################################
def Make_Rates(data, chgpts): #pass in the data and the changepoint locations
    chgpts = np.append(chgpts, [0, len(data)-1])#put zero and last point as beginning and ending cp
    nstates = len(chgpts)-1 #There are N-1 segments b/w cps
    chgpts = np.sort(chgpts) #sort the cps from 0 to npts-1
    #print(nstates)
    rates = np.zeros(nstates*nstates) #initialize the rate array
    sdev = np.zeros(nstates)#store std dev of each segment
    for index in range(nstates):
        cp1 = int(chgpts[index])
        cp2 = int(chgpts[index+1])
        rates[index] = np.mean(data[cp1:cp2])
        sdev[index] = max(np.std(data[cp1:cp2]), np.std(data)/4)#avoid a zero stdev
        for jj in range(nstates):
            if index==0:
                #do nothing
                continue
            elif jj==0 and index>0:
                rates[index*nstates+jj] = (nstates-index)/(0.05*len(data))  #decreasing reverse rates b/c CR (# dyes) is increasing
            elif jj==(nstates-1):
                rates[index*nstates+jj] = index/(0.05*len(data)) #increasing forward rates
            else:
                rates[index*nstates+jj] = 1e-6
#                rates[index*nstates+j] = index/(0.1*len(data))
	#endfor
#    rates[nstates:nstates**2] = 1/(0.1*len(data))#2*nstates/(0.1*len(data))
    #print(rates)
    return rates, sdev
######################End Make_Rates################################

######################## Prune_States ################################
## Take the rates created from Segment_Recurse and run Viterbi      ##
## If states are not acessed in the reconstruction, get rid of them ##
######################################################################
def Prune_States(data, rates, sdev):
    nstates = len(sdev)
    dx = 0.1
    clusters = np.zeros((nstates,2))
    count = np.array([], dtype=int)
    logP, HMM = logP_calc(data, rates, sdev, dx)#calculate the likelihood
    path = get_reconstruction(HMM, dx)
    for index in range(nstates):
        levelsFound = (path==index).sum()
        #Find how many times states are accessed
        if levelsFound > 0:
            clusters[index,0] = rates[index] #store the state's count rate
            clusters[index,1] = levelsFound #store the number of times accessed
        else:
            count = np.append(count, index)#keep track of unaccessed states
#    print(pd.DataFrame(clusters))
    if len(count) > 0:#delete unaccessed states
        clusters = np.delete(clusters, count, 0)
#    print(pd.DataFrame(clusters))
    #Now trim the rate and std dev arrays
    mask = np.ones(nstates**2, dtype=bool)
    new_nstates = int(len(clusters))
    mask[(new_nstates)**2:nstates**2+1] = False
    rates[0:new_nstates] = clusters[:,0]
    rates[new_nstates:nstates**2] = 1/(0.1*len(data))
    rates = rates[mask,...]
    temp_rates = rates[0:new_nstates]#get intensities only
    temp_rates, sdev = zip( *sorted( zip(temp_rates, sdev) ) )#sort count rates with sdevs following
    temp_rates = np.array(temp_rates)
    sdev = np.array(sdev)
    rates[0:new_nstates] = temp_rates #put the sorted intensities back
    return rates, sdev

######################## Prune_Unaccessed ############################
## Take the path created from logP_calc                             ##
## If states are not acessed in the reconstruction, get rid of them ##
######################################################################
def Prune_Unaccessed(path, rates, sdev):
    nstates = len(sdev)
    count = np.zeros(nstates, dtype=int)
    for index in range(nstates):
        levelsFound = (path==index).sum()
        #Find how many times states are accessed
        count[index] = levelsFound
#    print(count)
    #Now trim the rate and std dev arrays
    CR = rates[0:nstates]
    mask = np.array(count, dtype=bool)
    newCR = CR[mask,...]
    newsdev = sdev[mask,...]
    nstates = len(newCR)
    newrates = np.zeros(nstates**2)
    for index in range(nstates):
        newrates[index] = newCR[index]
        for jj in range(nstates):
            if index==0:
                #do nothing
                continue
            elif jj==0 and index>0:
                newrates[index*nstates+jj] = (nstates-index)/(0.05*len(path))  #decreasing reverse rates b/c CR (# dyes) is increasing
            elif jj==(nstates-1):
                newrates[index*nstates+jj] = index/(0.05*len(path)) #increasing forward rates
            else:
                newrates[index*nstates+jj] = 1e-6
    return newrates, newsdev

####################### Agglomerate ###############################
## Take a rate matrix from Segment_Recurse and attempt shrink it ##
###################################################################
def agglomerate(data, rates, sdev, target):
    nstates=int(np.sqrt(len(rates)))
    nst = nstates
    npts=int(len(data))
    dx=0.1
    clusters=np.zeros((nstates,2)) #a list of countrates and number of points they are accessed
    Wards = np.zeros((nstates, nstates))
    temp_rates = rates[0:nstates]#get intensities only
    temp_rates, sdev = zip( *sorted( zip(temp_rates, sdev) ) )#sort count rates with sdevs following
    temp_rates = np.array(temp_rates)
    sdev = np.array(sdev)
    rates[0:nstates] = temp_rates #put the sorted intensities back
    temp_rates = rates #store the full set of rates for safe keeping
#    print (sdev)
    logP, HMM = logP_calc(data, rates, sdev, dx)#calculate the likelihood
    BIC_prev = 2*logP+len(temp_rates)*np.log(npts)
#    print (f'starting BIC ={BIC_prev:.1f}')
#    print (f'nstates ={nstates:3d}')
    path = get_reconstruction(HMM, dx)
	#get Ward's minimum variance
    while(nst>target):#agglomerate to a target number of states
        rates_store = temp_rates
        sdev_store = sdev
        nstates=int(np.sqrt(len(temp_rates)))
        Wards = np.zeros((nstates, nstates))
        clusters = np.zeros((nstates,2))
        for index in range(nstates):
            levelsFound = (path==index).sum()
            #Find how many times states are accessed
            clusters[index,0] = temp_rates[index] #store the state's count rate
            clusters[index,1] = np.fmax(levelsFound, 1) #store the number of times accessed

        mask = np.ones(nstates**2, dtype=bool)
        nst = int(len(clusters))
        mask[(nst)**2:nstates**2+1] = False
        temp_rates[0:nst] = clusters[:,0]
        temp_rates = temp_rates[mask,...]
        Wards = np.zeros((nst, nst))
        #calculate distance between all pairs
        for x in range(nst):
            for y in range(nst):
                if clusters[x,1] > 0 and clusters[y,1] > 0:
                    Wards[x,y] = np.sqrt(2*clusters[x,1]*clusters[y,1]/(clusters[x,1]+clusters[y,1]))*(clusters[x,0]-clusters[y,0])**2
                else:
                    Wards[x,y] = (clusters[x,0]-clusters[y,0])**2
                if x==y: #diagonals are zero and we want to ignore them
                    Wards[x,y] = 1e300
        #Find the closest pair to try and combine into single
#        print(pd.DataFrame(Wards))
        minloc = np.argwhere(Wards == np.nanmin(Wards))#the row and column of the min
#        print(minloc)
#        print(f'old nstates {nstates:3d} new nstates {nstates-1:3d}')
#        print(f'combining intensities {temp_rates[minloc[0,0]]:.1f} and {temp_rates[minloc[0,1]]:.1f}')
        #weighted average of the count rates and std devs for combining
        ave_mins = (temp_rates[minloc[0,0]]*clusters[minloc[0,0],1]+temp_rates[minloc[0,1]]*clusters[minloc[0,1],1])/(clusters[minloc[0,0],1]+clusters[minloc[0,1],1])       
        term1 = np.fmax((clusters[minloc[0,0],1]-1), 0.01)*sdev[minloc[0,0]]**2
        term2 = np.fmax((clusters[minloc[0,0],1]-1),0.01)*sdev[minloc[0,1]]**2
        term3 = clusters[minloc[0,0],1]*clusters[minloc[0,1],1]*(temp_rates[minloc[0,0]]-temp_rates[minloc[0,1]])**2/(clusters[minloc[0,0],1]+clusters[minloc[0,1],1])
        term4 = clusters[minloc[0,0],1]+clusters[minloc[0,1],1]-1
 #       print(term1, term2, term3, term4)
        sdev_mins = np.sqrt((term1 + term2 + term3)/term4)
#        print("old rates", temp_rates[minloc[0,0]], temp_rates[minloc[0,1]], "new rate", ave_mins)
        temp_rates[minloc[0,0]] = ave_mins#insert ave CR of combined states
        sdev[minloc[0,0]] = sdev_mins #insert combined sdev
        mask = np.ones(len(temp_rates), dtype=bool)
        mask[minloc[0,1]] = False
        mask[(nst-1)**2+1:nst**2+1] = False
        temp_rates = temp_rates[mask,...]
        nst = int(np.sqrt(len(temp_rates)))
        for index in range(nst): #refill the transition rates properly
            for jj in range(nst):
                if index==0:
                    #do nothing
                    continue
                elif jj==0 and index>0:
                    temp_rates[index*nst+jj] = (nst-index)/(0.05*len(data))  #decreasing reverse rates b/c CR (# dyes) is increasing
                elif jj==(nstates-1):
                    temp_rates[index*nst+jj] = index/(0.05*len(data)) #increasing forward rates
                else:
                    temp_rates[index*nst+jj] = 1e-6
        sdev = np.delete(sdev, minloc[0,1])

	#End While

    return temp_rates, sdev

####################### End Agglomerate ###########################



########################## LogP_calc #############################
## Calculate the likelihood of a model given by a set of rates ###
## The log-likelihood calculator for an arbitrary number        ##
## of states. It has input of a time-series histogram trajectory##
## assuming 0.10 second binning.                                ##
##################################################################
def logP_calc(hist, kmat, sdev, binning):
    maxP = -1e300
    mm = int(np.sqrt(np.size(kmat)))
    nstates = int(np.sqrt(np.size(kmat)))
    ndim = np.size(kmat)
    npnts = np.size(hist)
    k = np.zeros(nstates) #store count rates
    sum_rmat = np.zeros(nstates)
    tempf = np.zeros(nstates)
    rmat = np.zeros(ndim)
    tmat = np.zeros(ndim)
    HMM = np.zeros((npnts, nstates), dtype='float64')
    HMM[0,:] = -1/nstates
#    print(kmat)
    for ii in range(nstates):
        k[ii] = kmat[ii]
#        print(k[ii])
        if k[ii]<=0:
            #print("bad emission rate")
            print(f'Bad emission rate {k[ii]:.1f}')
            return -1e300, HMM
        for jj in range(nstates): 
            if jj==ii:
                rmat[ii+nstates*jj] = 0
            else:
                rmat[ii+nstates*jj] = kmat[mm]
                sum_rmat[ii] += kmat[mm]
                if kmat[mm]<=0:    
                    print(f'Bad transition rate {rmat[ii+nstates*jj]:.1f}')
                    return -1e300, HMM
                mm+=1
    transprob = 0
    child = 0
    for ii in range(nstates):
            transprob = 1.0 - np.exp(-sum_rmat[ii]*binning) #short-time approximation to the master eqn
            for jj in range(nstates):
                if ii==jj:
                    tmat[ii+nstates*jj] = -sum_rmat[ii]*binning #log space no transition
                else:#if jj==ii-1 or jj==ii+1:
                    tmat[ii+nstates*jj] = np.log(rmat[ii+nstates*jj]*transprob/sum_rmat[ii]) #log space
#                else:
#                    tmat[ii+nstates*jj] = -1e-10
    #print(tmat)
    #recursion forward through data
    for ii in range(1, npnts):
        for jj in range(nstates):#child
            #mu = hist[ii]#int(hist[ii])
            #lamb = int(k[jj])
            #child = np.log(float(lamb**mu/factorial(mu)*np.exp(-lamb)))
            #child = np.log(1/(np.sqrt(2*np.pi)*sdev[jj]))-0.5*((hist[ii]-k[jj])/sdev[jj])**2
            child = norm.logpdf(hist[ii], loc=k[jj], scale=sdev[jj])
            #child = poisson.logpmf(lamb, mu)
            maxP = -1e300
            for aa in range(nstates):#parent
                tempf[aa] = HMM[ii-1,aa] + tmat[aa+nstates*jj]#find the most likely parent to the current state
                if tempf[aa] > maxP:
                    maxP = tempf[aa]
                    maxPfloor = np.floor(maxP)
            #Global adding up parents
            addlog = 0
            for dum in range(nstates):
                addlog += np.exp(tempf[dum] - maxPfloor)
            #endfor
            safelog = np.log(addlog) + maxPfloor + child
            HMM[ii, jj] = safelog
			#End global adding
            #HMM[ii,jj] = maxP
            #print(maxP)

    #recursion backward through data
    for ii in range(npnts-2, -1, -1):
        for jj in range(nstates):#child
            #mu = hist[ii]#int(hist[ii])
            #lamb = int(k[jj])
            #child = np.log(float(lamb**mu/factorial(mu)*np.exp(-lamb)))
            #child = np.log(1/(np.sqrt(2*np.pi)*sdev[jj]))-0.5*((hist[ii]-k[jj])/sdev[jj])**2
            child = norm.logpdf(hist[ii], loc=k[jj], scale=sdev[jj])
            #child = poisson.logpmf(lamb, mu)
            maxP = -1e300
            for aa in range(nstates):#parent
                tempf[aa] = HMM[ii+1,aa] + tmat[aa+nstates*jj]#find the most likely parent to the current state
                if tempf[aa] > maxP:
                    maxP = tempf[aa]
                    maxPfloor = np.floor(maxP)
            #Global adding up parents
            addlog = 0
            for dum in range(nstates):
                addlog += np.exp(tempf[dum] - maxPfloor)
            #endfor
            safelog = np.log(addlog) + maxPfloor + child
            HMM[ii, jj] = safelog
#            maxP = -1e300
#            for dum in range(nstates):
#                if HMM[ii, dum] > maxP:
#                    maxP = HMM[ii, dum]
			#End global adding
            #HMM[ii,jj] = maxP
    maxP = -1e300
    for dum in range(nstates):
        if HMM[ii, dum] > maxP:
            maxP = HMM[ii, dum]      
    return -maxP, HMM

########################## End LogP_calc #########################

#******************************************************************//
#Once we run the logp_calc, we can get a reconstruction from
#the likelihood matrix it creates called HMM
#*****************************************************************//
def get_reconstruction(HMM_matrix, binning):
    shape = np.shape(HMM_matrix)
    nstates = shape[1] #This isn't right
    npnts = shape[0]
    path = np.zeros(npnts)
    maxP = -1e300
    for ii in range(npnts):
        maxP = -1e300
        for jj in range(nstates):
            if HMM_matrix[ii,jj] > maxP:
                maxP = HMM_matrix[ii,jj]
                path[ii] = jj
            elif HMM_matrix[ii,jj] == maxP:
                path[ii]=path[ii-1]
    return path
####### END get_reconstruction #####################################

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(parent=root)
root.destroy()
df = pd.read_csv(file_path, header=None, index_col=0, delim_whitespace=True, dtype=np.float64).T
binaries = df.columns.values #The first column is binary pre-selection from Song's ML
names = list(range(len(binaries)))
#print(names)
df.columns=names
print(df.head())

index = 0
binning = 0.1
path_df = pd.DataFrame()#empty dataframe to store path results
rate_df = pd.DataFrame()#empty dataframe to store optimized rates
sdev_df = pd.DataFrame()#empty dataframe to store optimized rates
result_hist = np.zeros(3)
for col in df.columns:
    if binaries[index]==0:#We only want pre-sort binary value of 1
        index += 1
        continue
#    if index > 1:
#        break
    print(f'working on {col}')
    df.index = range(len(df))
    temp_hist = df[col] #temporarily copy the current data
    npts = np.size(temp_hist)
    cp = np.array([])
    #Recursively find segments
    StudT, cp = Segment_Recurse(temp_hist, 0, npts) #Find statistically different segments
    #create a rate array
    rmat, sdevs = Make_Rates(temp_hist, cp)
    nstates = int(np.sqrt(len(rmat)))
    nstates_hi = nstates
    print(f'Segmenter found {nstates:3d} segments.')
    #agglomerate the segments
    BIC = np.zeros(3)
    BICmin = 1e300
    keeprmat = rmat
    keepsdev = sdevs
    for ii in range(4, 1, -1):#try 3, 2, 1 pb steps
        rmat, sdevs = agglomerate(temp_hist, rmat, sdevs, ii)
        #print("exiting agglomeration, nstates =", np.sqrt(len(rmat)))
        logP, HMM = logP_calc(temp_hist, rmat, sdevs, binning)#calculate the likelihood
        if logP==1e300:#Something didn't work to improve model
            rmat = keeprmat
            sdevs = keepsdev
            continue
        path = get_reconstruction(HMM, binning)
        ########### trying to prune ###################
        rmat, sdevs =Prune_Unaccessed(path, rmat, sdevs) #get rid of unaccessed states
        logP, HMM = logP_calc(temp_hist, rmat, sdevs, binning)#calculate the likelihood
        if logP==1e300:#Something didn't work to improve model
            rmat = keeprmat
            sdevs = keepsdev
            continue
        path = get_reconstruction(HMM, binning)
        ################## end pruning ################
        BIC[ii-2] = 2*logP+ii**2*np.log(npts)
        print("nstates", ii, "BIC", BIC[ii-2])
        if BIC[ii-2] < BICmin:
            BICmin = BIC[ii-2]
            keeprmat = rmat
            keepsdev = sdevs

    plt.plot(np.arange(2, 5, 1), BIC, '-o')
    plt.xlabel('Number of states')
    plt.ylabel('BIC')
    plt.savefig('./figures/BIC_'+str(col)+'.png')
    plt.show()
    rmat = keeprmat
    sdevs = keepsdev
    nstates = int(np.sqrt(len(rmat)))
    logP, HMM = logP_calc(temp_hist, rmat, sdevs, binning)#calculate the likelihood
    path = get_reconstruction(HMM, binning)
    BIC = 2*logP+nstates**2*np.log(npts)
#    print (f'ending BIC ={BICmin:.1f}')
#    print (f'best nstates ={nstates:3d}')
    print ("rates =", rmat)
    print("sdev = ", sdevs)
    result_hist[nstates-2] += 1
    #Assemble dataframes to save
    path = pd.DataFrame(path, columns = [str(col)+'_path'])
    path_df = pd.concat([path_df, path], ignore_index=False, axis=1)
    rmat = pd.DataFrame(rmat, columns = [str(col)+'_rmat'])
    rate_df = pd.concat([rate_df, rmat], ignore_index=False, axis=1)
    sdevs = pd.DataFrame(sdevs, columns = [str(col)+'_sdev'])
    sdev_df = pd.concat([sdev_df, sdevs], ignore_index=False, axis=1)
    #Plot the result
    timeax = np.arange(0, len(temp_hist)*0.1, 0.1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('intensity', color=color)
    ax1.plot(timeax,temp_hist, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('path', color=color)  # we already handled the x-label with ax1
    ax2.plot(timeax, path, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('./figures/Reconstruct_'+str(col)+'.png')
    plt.show()
    

    index += 1
plt.bar(np.arange(1, 4, 1), result_hist)
plt.xlabel('number of GFP')
plt.ylabel('number of observations')
plt.savefig('./figures/StoichTotal_'+str(col)+'.png')
plt.show()
result_hist = pd.DataFrame(result_hist, columns = ['FinalHist'])
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('./Optimized_Rates.xlsx', engine='xlsxwriter')
path_df.to_excel('./Reconstructed_paths.xlsx')
rate_df.to_excel(writer, sheet_name='rates')
sdev_df.to_excel(writer, sheet_name='stdevs')
result_hist.to_excel(writer, sheet_name='FinalHist')
writer.save()