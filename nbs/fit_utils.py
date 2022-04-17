from pathlib import Path
from matplotlib import pyplot as plt
from  matplotlib.ticker import MultipleLocator
import peakutils
import pandas as pd
import numpy as np
from scipy import stats
from lmfit import Parameters, minimize

def PseudoVoigtFunction(WavNr, Pos, Amp, GammaL, FracL):
        SigmaG = GammaL / np.sqrt(2*np.log(2)) # Calculate the sigma parameter  for the Gaussian distribution from GammaL (coupled in Pseudo-Voigt)
        LorentzPart = Amp * (GammaL**2 / ((WavNr - Pos)**2 + GammaL**2)) # Lorentzian distribution
        GaussPart = Amp * np.exp( -((WavNr - Pos)/SigmaG)**2) # Gaussian distribution
        Fit = FracL * LorentzPart + (1 - FracL) * GaussPart # Linear combination of the two parts (or distributions)
        return Fit

def one_pv(pars, x, data=None, eps=None): #Function definition
    # unpack parameters, extract .value attribute for each parameter
    a3 = pars['a3'].value
    c3 = pars['c3'].value
    s3 = pars['s3'].value
    f3 = pars['f3'].value

    peak1 = PseudoVoigtFunction(x.astype(float),c3, a3, s3, f3)

    model =  peak1  # The global model is the sum of the Gaussian peaks

    if data is None: # if we don't have data, the function only returns the direct calculation
        return model, peak1
    if eps is None: # without errors, no ponderation
        return (model - data)
    return (model - data)/eps # with errors, the difference is ponderated

def three_pv(pars, x, data=None, eps=None): #Function definition
    # unpack parameters, extract .value attribute for each parameter
    a1 = pars['a1'].value
    c1 = pars['c1'].value
    s1 = pars['s1'].value
    f1 = pars['f1'].value
    
    a4 = pars['a4'].value
    c4 = pars['c4'].value
    s4 = pars['s4'].value
    f4 = pars['f4'].value
    
    a2 = pars['a2'].value
    c2 = pars['c2'].value
    s2 = pars['s2'].value
    f2 = pars['f2'].value

    peak1 = PseudoVoigtFunction(x.astype(float), c1, a1, s1, f1)
    peak3 = PseudoVoigtFunction(x.astype(float), c4, a4, s4, f4)
    peak2 = PseudoVoigtFunction(x.astype(float), c2, a2, s2, f2)

    model =  peak1 + peak3 + peak2  # The global model is the sum of the Gaussian peaks

    if data is None: # if we don't have data, the function only returns the direct calculation
        return model, peak1, peak3, peak2
    if eps is None: # without errors, no ponderation
        return (model - data)
    return (model - data)/eps # with errors, the difference is ponderated

def baseline_data(df1: pd.DataFrame, df2: pd.DataFrame, \
                df3: pd.DataFrame, df4: pd.DataFrame) -> pd.DataFrame:
    """Baseline data for fitting.
        df1: foreground1D
        df2: foreground2D
        df3: background1D
        df4: background2D
    """
    df1['I'] = df1['I']-df3['I']
    base1 = peakutils.baseline(df1['I'], 1)
    df1['I_base']= df1['I']-base1
    df1 = df1[(df1['W']>1220) & (df1['W']<1750)]

    print('\nBASELINED 1500')

    df2['I'] = df2['I']-df4['I']
    df2 = df2[(df2['W']>2550) & (df2['W']<2850)]
    df2= df2[(np.abs(stats.zscore(df2))<3).all(axis=1)]
    base2 = peakutils.baseline(df2['I'], 1)
    df2['I_base'] = df2['I']-base2

    print('BASELINED 2700')

    return df1, df2

def fit_data(df1: pd.DataFrame, df2: pd.DataFrame) -> minimize:
    """Fitting data taken from:
        df1: foreground1D
        df2: foreground2D
    """
    
    print ('ADDING PARAMS')

    ps1 = Parameters()

    #            (Name,  Value,  Vary,   Min,  Max,  Expr)
    ps1.add_many(('a1',    1 ,   True,     0, None,  None),
                 ('c1',   1350,   True,  1330, 1370,  None),
                 ('s1',     20,   True,    10,   200,  None),  # 200 so that we get proper fit width of unpatterned peak 
                 ('f1',    0.5,   True,  0, 1,  None),
                 ('a4',    1 ,   True,     0, None,  None), # peak middle of GD
                 ('c4',   1500,   True,  1480, 1520,  None),
                 ('s4',     20,   True,    10,   200,  None),  
                 ('f4',    0.5,   True,  0, 1,  None),
                 ('a2',      1,   True,     0, None,  None),
                 ('c2',    1600,   True, 1560,  1640,  None),
                 ('s2',     20,   True,    10,   200,  None),
                 ('f2',    0.5,   True,  0, 1,  None))

    ps2 = Parameters()

    #            (Name,  Value,  Vary,   Min,  Max,  Expr)
    ps2.add_many(('a3',      1,   True,     0, None,  None),
                 ('c3',    2700,   True, 2650,  2750,  None),
                 ('s3',     20,   True,    10,   200,  None),
                 ('f3',    0.5,   True,  0, 1,  None))
    print('FITTING SPECTRA')
    x, y = df1['W'], df1['I_base']
    out1 = minimize(three_pv, ps1, method = 'leastsq', args=(x, y))

    x2, y2 = df2['W'], df2['I_base']
    out2 = minimize(one_pv, ps2, method = 'leastsq', args=(x2, y2))

    return out1, out2

def plot_data(df1: pd.DataFrame, df2: pd.DataFrame, out1: minimize, out2: minimize, filename: Path) -> None:
    """Plotting data taken from:
        df1: foreground1D
        df2: foreground2D
        out1: fitting result of foreground1D
        out2: fitting result of foreground2D
    """
    print('PLOTTING DATA')
    x, y = df1['W'], df1['I_base']
    x2, y2 = df2['W'], df2['I_base']

    f, (ax,ax2)=plt.subplots(1,2,sharey=True, gridspec_kw = {'width_ratios':[2.5, 1]})
    f.subplots_adjust(wspace=0.1)

    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax2.xaxis.set_major_locator(MultipleLocator(200))

    ax.set_yticklabels([])

    ax.plot(x,y,'-', label='measured',)
    ax.plot(x,three_pv(out1.params, x)[0], label='fit')
    ax.plot(x,three_pv(out1.params, x)[1], label='fit')
    ax.plot(x,three_pv(out1.params, x)[2], label='fit')
    ax.plot(x,three_pv(out1.params, x)[3], label='fit')
#     ax.plot(x,four_pv(out.params, x)[4], label='fit')
    ax2.plot(x2,y2,'-')
    ax2.plot(x2,one_pv(out2.params, x2)[0])

    f.text(0.05, 0.5, 'Intensity [a.u.]', va='center', rotation='vertical', fontsize=16)
    f.text(0.5, 0.01, 'Raman shift [cm$^{-1}$]', ha='center', rotation='horizontal',fontsize=16)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')  # don't put tick labels at the top
    ax2.yaxis.tick_right()
    # ax.yaxis.label('test')

    d = .02  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, + d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((- d, + d), (- d, + d), **kwargs)  # bottom-left diagonal
    ax2.plot((- d, + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # ax.legend(loc='upper right')
#     plt.savefig(p/'Raman_raw_111.png', format='png', dpi=300)
    #plt.show()
    save_file = filename.stem + ".png"
    plt.savefig(save_file, dpi=200)
    print('SAVED FILE')

def extract_data(out1: minimize, out2: minimize,
                df: pd.DataFrame, filename: Path, target: str = 'GD') -> float:
    print(f'EXTRACTING DATA FOR {target}')
    
    d1 = pd.DataFrame({key: [par.value] for key, par in out1.params.items()})
    d2 = pd.DataFrame({key: [par.value] for key, par in out2.params.items()})

    d = pd.concat([d1,d2],axis=1)
    

    if d['s1'].values > 300:
        d[['a1','c1','s1','f1']] = 0

    if d['s2'].values > 120:
        d[['a2','c2','s2','f2']] = 0

    if d['s3'].values > 120:
        d[['a3','c3','s3','f3']] = 0
        
    d.columns= ['D','PD','WD','FD','D1','PD1','WD1','FD1','G','PG','WG','FG','2D','P2D','W2D','F2D']
   
    d['GD']  = d['G'] / d['D']
    d['2DG'] = d['2D'] / d['G']

    d['filename'] = filename.stem
    d.to_csv('fit_results.csv', index=False, mode='a', header=False)

    if d['WD'].values > 120:
        if (d['D'].values>.3*d['G'].values or d['D1'].values > d['D'].values):
            print("D-width > 120: patterning not done")
    elif (d['WG'].values > 120):
        print("G-width > 120: patterning not done")
        d3 = pd.read_csv('dataset.csv')
        d3['ratio'].replace(' ', np.nan, inplace = True)
        d4 = d3.dropna(subset = ["ratio"])
        a = d4['ratio'].shape
        d3.loc[a[0],'ratio'] = 0
        # d3.to_csv('dataset.csv', index = False)
    
    
    elif np.mean(df[df['W']<1255]['I_base']) > 0.7*np.mean(df[(df['W']>1340) & (df['W']<1350)]['I_base']) \
        or np.mean(df[(df['W']>1400) & (df['W']<1550)]['I_base']) > 0.7*np.mean(df[(df['W']>1340) & (df['W']<1350)]['I_base']) \
        and (d['GD'].values[0] <= 1.2):
        print("Intensities @ 1255, 1500 too high: patterning not done")
    
    else:

        d3=pd.read_csv('dataset.csv')
        d3['ratio'].replace(' ',np.nan, inplace=True)
        d4=d3.dropna(subset=["ratio"])
        a=d4['ratio'].shape
        if target == '2DG':
            d3.loc[a[0],'ratio'] = d['2DG'].values[0]
        else:
            d3.loc[a[0],'ratio'] = d['GD'].values[0]
        # d3.to_csv('dataset.csv',index=False)
    return d['2DG'].values[0] if target == '2DG' else d['GD'].values[0]