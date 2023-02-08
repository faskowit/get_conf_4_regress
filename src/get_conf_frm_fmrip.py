# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import json
import argparse


NUSCHOICES= ["36P", "9P", "6P", 
             "aCompCor", "24aCompCor", "24aCompCorGsr",
             "globalsig", "globalsig4", "linear",
             "2phys", "2physGsr", "8phys" ]


def get_spikereg_confounds(motion_ts, threshold):
    """
    motion_ts = [0.1, 0.7, 0.2, 0.6, 0.3]
    threshold = 0.5
    get_spikereg_confounds(motion_ts, threshold)

    returns
    1.) a df with spikereg confound regressors (trs with motion > thershold)
       outlier_1  outlier_2
    0          0          0
    1          1          0
    2          0          0
    3          0          1
    4          0          0

    2.) a df with counts of outlier and non-outlier trs
       outlier  n_tr
    0    False     3
    1     True     2

    note, from Ciric et. al, "the conversion of FD to RMS displacement is approximately 2:1"...
    -> here we are using FD for spike thr, so a value of 0.5 is ~ to the 0.25mm RMS spike thr of 36P method

    """
    df = pd.DataFrame({"motion": motion_ts})
    df.fillna(value=0, inplace=True)  # first value is nan
    df["outlier"] = df["motion"] > threshold
    outlier_stats = df.groupby("outlier").count().reset_index().rename(columns={"motion": "n_tr"})

    df["outliers_num"] = 0
    df.loc[df.outlier, "outliers_num"] = range(1, df.outlier.sum() + 1)
    outliers = pd.get_dummies(df.outliers_num, dtype=int, drop_first=True, prefix="outlier")

    return outliers, outlier_stats, df


def get_confounds(confounds_file, kind="36P", spikereg_threshold=None, 
                  confounds_json='', dctbasis=False, addreg='', initdum=0,
                  addlin=False,censor_file=False):
    """
    takes a fmriprep confounds file and creates data frame with regressors.
    kind == "36P" returns Satterthwaite's 36P confound regressors
    kind == "9P" returns CSF, WM, Global signal + 6 motion parameters (used in 
            Ng et al., 2016)
    kind == "aCompCor"* returns model no. 11 from Parkes
    kind == "24aCompCor"* returns model no. 7 from Parkes
    kind == "24aCompCorGsr"* returns model no. 9 from Parkes

    if spikereg_threshold=None, no spike regression is performed

    Satterthwaite, T. D., Elliott, M. A., Gerraty, R. T., Ruparel, K., 
    Loughead, J., Calkins, M. E., et al. (2013). An improved framework for 
    confound regression and filtering for control of motion artifact in the 
    preprocessing of resting-state functional connectivity data. NeuroImage, 
    64, 240?256. http://doi.org/10.1016/j.neuroimage.2012.08.052

    Parkes, L., Fulcher, B., Yücel, M., & Fornito, A. (2018). An evaluation
    of the efficacy, reliability, and sensitivity of motion correction
    strategies for resting-state functional MRI. NeuroImage, 171, 415-436.

    """

    df = pd.read_csv(confounds_file, sep="\t")

    # check if old/new confound names
    p6cols = ''
    p9cols = ''
    if 'GlobalSignal' in df:
        print("detected old confounds names")
        # imgsignals = ['CSF', 'WhiteMatter', 'GlobalSignal']
        p6cols = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
        p9cols = ['CSF', 'WhiteMatter', 'GlobalSignal', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
        globalsignalcol = ['GlobalSignal']
        compCorregex = 'aCompCor'
        framewisecol = 'FramewiseDisplacement'
        phys2cols = ['CSF', 'WhiteMatter']

    elif 'global_signal' in df:
        print("detected new confounds names")
        # imgsignals = ['csf', 'white_matter', 'global_signal']
        p6cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        p9cols = ['csf', 'white_matter', 'global_signal', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        globalsignalcol = ['global_signal']
        compCorregex = 'a_comp_cor_'
        framewisecol = 'framewise_displacement'
        phys2cols = ['csf', 'white_matter'] 

    else:
        print("trouble reading necessary columns from confounds file. exiting")
        exit(1)

    # extract nusiance regressors for movement + signal
    p6 = df[p6cols]
    p9 = df[p9cols]

    # 6Pder
    p6_der = p6.diff().fillna(0)
    p6_der.columns = [c + "_der" for c in p6_der.columns]
    
    # 9Pder
    p9_der = p9.diff().fillna(0)
    p9_der.columns = [c + "_der" for c in p9_der.columns]
    
    # 12P
    p12 = pd.concat((p6, p6_der), axis=1)
    p12_2 = p12 ** 2
    p12_2.columns = [c + "_2" for c in p12_2.columns]
    
    # 18P + 18P^2
    p18 = pd.concat((p9, p9_der), axis=1)
    p18_2 = p18 ** 2
    p18_2.columns = [c + "_2" for c in p18_2.columns]
    
    # 36P
    p36 = pd.concat((p18, p18_2), axis=1)

    # GSR4
    gsr = df[globalsignalcol]
    gsr2 = gsr ** 2
    gsr_der = gsr.diff().fillna(0)
    gsr_der2 = gsr_der ** 2
    gsr4 = pd.concat((gsr, gsr2, gsr_der, gsr_der2), axis=1)

    # 2phys
    phy2 = df[phys2cols]
    phy2gsr = pd.concat((phy2,df[globalsignalcol]),axis=1)

    phy2_der = phy2.diff().fillna(0)
    phy2_der.columns = [c + "_der" for c in phy2_der.columns]
    
    phy4 = pd.concat((phy2,phy2_der), axis=1)
    
    phy4_2 = phy4 ** 2 
    phy4_2.columns = [c + "_2" for c in phy4_2.columns]

    # phy8    
    phy8 = pd.concat((phy4,phy4_2), axis=1)

    if kind == "globalsig":
        confounds = gsr
    elif kind == "globalsig4":
        confounds = gsr4
    elif kind == "36P":
        confounds = p36
    elif kind == "12P":
        confounds = p12 ;
    elif kind == "9P":
        confounds = p9
    elif kind == "6P":
        confounds = p6
    elif kind == "2phys":
        confounds = phy2 
    elif kind == "2physGsr":
        confounds = phy2gsr
    elif kind == "8phys":
        confounds = phy8
    elif kind == "linear":
        pass
    else:
        # then we grab compcor stuff
        # get compcor nuisance regressors and combine with 12P
        aCompC = df.filter(regex=compCorregex)
        if aCompC.empty:
            print("could not find compcor columns. exiting")
            exit(1)
        elif aCompC.shape[1] > 10:

            # if the confounds json is available, read the variance explained
            # from the 'combined' 'Mask' components, and use top 5 of those
            if confounds_json:
                # read the confounds json
                with open(confounds_json, 'r') as json_file:
                    confjson = json.load(json_file)
                print('read confounds json')

                # initalize lists
                combokeys = []
                varex = []
                for key in confjson:
                    if 'Mask' in confjson[key].keys():
                        if confjson[key]['Mask'] == 'combined':
                            combokeys.append(key)
                            varex.append(confjson[key]['VarianceExplained'])

                # get the sort based on variance explained
                sortvar = np.argsort(varex)
                aCCcolnames = [combokeys[i] for i in sortvar[-5:]]
                aCompC = aCompC[aCCcolnames]

            else:
                # if there are more than 5 columns, take only the first five components
                aCCcolnames = [(''.join([compCorregex, "{:0>2}".format(n)])) for n in range(0, 5)]
                aCompC = aCompC[aCCcolnames]

        p12aCompC = pd.concat((p12, aCompC), axis=1)
        p24aCompC = pd.concat((p12, p12_2, aCompC), axis=1)

        if kind == "aCompCor":
            confounds = p12aCompC
        elif kind == "24aCompCor":
            confounds = p24aCompC
        elif kind == "24aCompCorGsr":
            confounds = pd.concat((p24aCompC, gsr4), axis=1)
        elif kind == "linear":
            pass
        else:
            # it will never get here, but assign confounds so my linter doesn't complain
            confounds = ''
            exit(1)

    # linear column
    if kind != "linear" and addlin:
        # high pass filter should take care of most linear trend... 
        # could remove rest of linear trend in the makemat function
        confounds['lin'] = list(range(1, confounds.shape[0]+1)) 
    elif kind == "linear" : # it is "linear"
        confounds = pd.DataFrame(list(range(1, df.shape[0]+1)))

    # if using dctbasis, get these from confounds file and add it
    if dctbasis:
        cosconfounds = df.filter(regex='cosine')
        confounds = pd.concat((confounds, cosconfounds), axis=1)

    # any additional regressors to add?
    if addreg:
        addregtable = pd.read_csv(addreg, sep="\t")
        confounds = pd.concat([confounds, addregtable], axis=1) 

    # and dummy regressors at beginning to add?
    if initdum:
        ii = np.zeros(confounds.shape[0])
        # add vals
        ii[0:initdum] = np.arange(1,(initdum+1))
        dumpd = pd.get_dummies(ii,drop_first=True,prefix='initdum')
        confounds = pd.concat([confounds, dumpd], axis=1) 
    
    # outlier stuff
    cen_df = None
    outlier_stats = None 
    if spikereg_threshold:
        outliers, outlier_stats, spike_df = get_spikereg_confounds(df[framewisecol].values, spikereg_threshold)
        if censor_file:
            cen_df = pd.DataFrame(np.int8(spike_df['outlier']==False)) 
        else:
            confounds = pd.concat([confounds, outliers], axis=1)

    return confounds, outlier_stats, cen_df


def main():
    
    parser = argparse.ArgumentParser(description='get coonfounds')

    parser.add_argument('confounds', type=str, help='input confounds file (from fmriprep)')
    parser.add_argument('-out', type=str, help='ouput base name',
                        default='output')    

    # choices
    parser.add_argument('-strategy', type=str, help='confound strategy',
                        choices=NUSCHOICES,
                        default='36P')
    parser.add_argument('-spikethr', type=float, help='spike threshold value',
                        default=None)
    parser.add_argument('-initaldummy', type=int, help='add x regressors to beginning of data',
                        choices=range(1, 50))
    
    # other inputs
    parser.add_argument('-confjson', type=str, help='confound json file, output by newer version of fmriprep',
                        default=None)
    parser.add_argument('-add_regressors', type=str, help='add these regressors')

    # flags    
    parser.add_argument('-add_dct', help='add the dct basis provided by fmriprep',
                        action='store_true')
    parser.add_argument('-add_linear', help='add linear regressor',
                        action='store_true')
    parser.add_argument('-cen_out_file', help='output an afni-style censor file',
                        action='store_true')

    # parse
    args = parser.parse_args()

    # print the args
    print("\nARGS: ")
    for arg in vars(args):
        print("{} {}".format(str(arg), str(getattr(args, arg))))
    print("END ARGS\n")

    # run eeeeet
    # def get_confounds(confounds_file, kind="36P", spikereg_threshold=None, 
    #                  confounds_json='', dctbasis=False, addreg='', initdum=0,
    #                  addlin=False):
    [outconf,outdfstat,outcen] = get_confounds(args.confounds, args.strategy,
                                        args.spikethr, args.confjson, 
                                        args.add_dct, args.add_regressors, 
                                        args.initaldummy, args.add_linear,
                                        args.cen_out_file)

    # write it out
    outconf.to_csv(''.join([args.out, '_conf.csv']),index=False,float_format='%.8g')

    if outdfstat is not None:
        outdfstat.to_csv(''.join([args.out, '_outlierstat.csv']),index=False)

    if outcen is not None:
        outcen.to_csv(''.join([args.out, '_cen.csv']),index=False,header=False)


if __name__ == '__main__': # run it
    main()
