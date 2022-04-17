import pandas as pd
import numpy as np
import os,csv
from pathlib import Path
from random import randrange
from datetime import date

def write_data_files(path: Path, series: int, nr_random_lines: int = 7, \
    power_range: set = (5,1190), time_range: set = (1050,5000), pressure_range: set = (100,350)) -> None:
        """ Writes data files with the parameter design at path:
            path: Path of the campaign
            series: Campaign number
            nr_random_lines: Number of starting lines, defaults at 7
            power_range: set of minPower, maxPower, default (5, 1190)
            time_range: set of minTime, maxTime, default (1050, 5000)
            pressure_range: set of minPressure, maxPressure, default (100,350)
        """
        power=[]
        line_time=[]
        pressure=[]

        print("TODAY's DATE:",str(date.today()))

        for x in range(nr_random_lines):
                powr = randrange(power_range[0], power_range[1], 1) # in mW
                tm = randrange(time_range[0], time_range[1], 1) # in ms
                pr = randrange(pressure_range[0], pressure_range[1], 10) # in psi
        for i in range(9):
                power.append(powr)
                line_time.append(tm)
                pressure.append(pr)

        p=str(date.today())+"-Series-"+str(series)
        print(f"Here is the new Campaign Folder:{p}")
        # path = Path(r'c:\\Users\\UWAdmin\\Desktop\\_pyControl\\campaigns')
        os.chdir(path)
        os.mkdir(p)
        os.chdir(p)

        row=['power','time','pressure','ratio']
        plot_row=['power','time','pressure','dummy_ratio', 'pred_mean', 'pred_upper', 'ei']
        #rename this to raw_post_processed.csv
        with open('dataset.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                writer.writerows(zip(power, line_time, pressure))

        #rename this to raw_pre-patterning.csv
        with open('dataset-pre.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                writer.writerows(zip(power, line_time, pressure))

        # THIS HAS THREE ADDITIONAL COLUMNS
        with open('plot_data.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(plot_row)
                # writer.writerows(zip(power, line_time, pressure))

        df2=pd.read_csv('dataset.csv')
        df2=df2.drop_duplicates() #keep only the unique rows
        df2.head()
        df2.to_csv('data.csv',index=False) #this is what will be read by mlrMBO in th R code

def write_more() -> None:
    """
    Used after doing the AI stuff
    """
    d = pd.read_csv('data.csv')
    ln = d.shape[0]

    vpower = d['power'][ln-1]
    vtime = d['time'][ln-1]
    vpressure = d['pressure'][ln-1]

    d1 = pd.read_csv('dataset.csv')
    ln = d1.shape[0]
    d1.loc[ln,"power"] = vpower
    d1.loc[ln,"time"] = vtime
    d1.loc[ln,"pressure"] = vpressure
    d1.to_csv('dataset.csv',index = False)
    d1.to_csv('dataset-pre.csv',index = False)
def repeats() -> None:
    # d1 = pd.read_csv('data.csv')
    df2 = pd.read_csv('dataset.csv')
    ln = len(df2['power'])
    m = ln
    print(f"DATASET LENGTH: {ln}")
    counter  = m
    for i in range(8):
        toAdd = [df2['power'][m-1],df2['time'][m-1],df2['pressure'][m-1]]
        filename = "dataset.csv"
        with open(filename, "r") as infile:
            reader = list(csv.reader(infile))
            reader.insert(counter+1, toAdd)

        with open(filename, "w", newline='') as outfile:
            writer = csv.writer(outfile)
            for line in reader:
                writer.writerow(line)
                
    for i in range(8):
        toAdd = [df2['power'][m - 1],df2['time'][m - 1],df2['pressure'][m - 1]]
        filename = "dataset-pre.csv"
        with open(filename, "r") as infile:
            reader = list(csv.reader(infile))
            reader.insert(counter + 1, toAdd)

        with open(filename, "w", newline='') as outfile:
            writer = csv.writer(outfile)
            for line in reader:
                writer.writerow(line)

def take_mean(save_line: int, spots_measured: int) -> float:

    print(os.getcwd())
    df = pd.read_csv('dataset.csv')
    result = df['ratio'].iloc[-spots_measured:].mean()
    df['ratio'].iloc[-spots_measured:].fillna(0, inplace=True)
    df.to_csv('dataset.csv', index = False)

    df2 = pd.read_csv('data.csv')
    df2.loc[save_line,"ratio"] = result
    df2.to_csv('data.csv',index = False)
    return result

def get_move_y(lines: int, start_y: int, step_y: int = 1) -> list:
    move_y = [0 for i in range(int(lines))]
    for i in range(int(lines)):
    #     print(i)
        if i == 0: move_y[i] = float(start_y)
        if i > 0 :
            move_y[i] = float(move_y[i-1]) + float(step_y)
    return move_y

# def take_mean(steps: float, save_line: int) -> float:
    
#     print(os.getcwd())
#     df = pd.read_csv('dataset.csv')

#     valss = np.sort([df['ratio'][steps+8], df['ratio'][steps+7], df['ratio'][steps+6],
#                     df['ratio'][steps+5], df['ratio'][steps+4], df['ratio'][steps+3],
#                    df['ratio'][steps+2], df['ratio'][steps+1], df['ratio'][steps]])

#     lst = [s for s in valss if str(s) != 'nan']
#     result = np.mean(lst)
#     df2 = pd.read_csv('data.csv')
#     df2.loc[save_line,"ratio"] = result
#     df2.to_csv('data.csv',index = False)
#     return result