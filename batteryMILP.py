
import numpy as np
import pandas as pd
from math import isclose
import os
from pulp import *
import matplotlib.pyplot as plt


timeLine, windGenXlsx = 8760, r"gpwind2021gross.xlsx"

class BaseModel(object):

    def __init__(self,
                 fileName : str,
                 searchPath : str = r"C:\\Users\\" + str(os.getlogin()) + "\\Desktop\\batteryData",
                 battery_capacity : float = 200.0,
                 initialSOC : float = 100.0,
                 cThreshold : float = .25,
                 dThreshold : float = .95,
                 batteryPower : float = 50.0,
                 max_charge_rate : float = 50.0,
                 max_discharge_rate : float = 50.0,
                 max_daily_throughput : float = 200.0,
                 hours : range = range(24),
                 firm_block_limit : float = 100.0,
                 results : list = [],
                 sellP : float = 300.0,
                 purchaseP : float = 300.0,
                 cLoss : float = .07,
                 dLoss : float = .07,
                 yearCycles : int = 365,
                 dailyCycles : int = 1,
                 firstHour : int = 0,
                 start_hour : int = 15,
                 end_hour : int = 22,
                 minSOC : int = 0) -> None:
        self.fileName, self.searchPath, self.battery_capacity, self.initialSOC, self.cThreshold, \
            self.dThreshold, self.batteryPower, self.max_charge_rate, self.max_discharge_rate, \
                self.max_daily_throughput, self.hours, self.firm_block_limit, self.results, \
                    self.sellP, self.purchaseP, self.cLoss, self.dLoss, self.yearCycles, \
                        self.dailyCycles, self.firstHour, self.start_hour, self.end_hour, \
                             self.minSOC = fileName, searchPath, \
                                battery_capacity, initialSOC, cThreshold, dThreshold, \
                                    batteryPower, max_charge_rate, max_discharge_rate, \
                                        max_daily_throughput, hours, firm_block_limit, \
                                            results, sellP, purchaseP, cLoss, dLoss, yearCycles, \
                                                dailyCycles, firstHour, start_hour, end_hour, minSOC



    def queryFile(self) -> str:
        for root, _, files in os.walk(self.searchPath):
            return os.path.join(root, self.fileName) \
                if self.fileName in files else None



    def getSOC(self,
               energyTransfered : float,
               asPercentage : bool = False) -> float:
        r"""
        Returns the battery state of charge (SOC).
        If asPercentage is set to True the return represents the percentage of the maximum battery capacity (%)
        else it represents the energy stored in the battery in (MWh)

        energyTransfered --> Amount of energy charging or discharging the battery resource (MWh)
        + : charging
        - : discharging
        """
        return (self.initialSOC + energyTransfered/self.battery_capacity) * 100 if asPercentage \
            else (self.initialSOC + energyTransfered/self.battery_capacity)




class ImportData(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def loadData(self) -> pd.DataFrame:
        r"Returns the load consumption dataframe in MWh"
        pass


    def windData(self) -> pd.DataFrame:
        r"Returns the Wind Energy dataframe in MWh"
        windDf = pd.read_excel(self.queryFile())
        return windDf


    def solarData(self) -> pd.DataFrame:
        r"Returns the Solar Energy dataframe in MWh"
        pass


    def sellingPrice(self) -> pd.DataFrame:
        r"Returns the selling price dataframe (to sell to the grid) in currency/MWh"
        return self.sellP


    def purchasingPrice(self) -> pd.DataFrame:
        r"Returns the electricity price (to purchase from the grid) dataframe in currency/MWh"
        return self.purchaseP



class OptimizeModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def findMinVal(self,
                   inptArr : list) -> int:
        minVal = inptArr[0]
        for num in inptArr[1:]:
            if num < minVal:
                minVal = num
        return minVal


    def adjustCol(self,
                  inptDf : pd.DataFrame,
                  colName : str = "SoC(MWh)") -> pd.DataFrame:

        # TAKE THE SOC COLUMN AS DATAFRAME (df)
        df = inptDf[colName].to_frame()

        prevVal = None
        for i, row in df.iterrows():

            rowLst = list(row.values)
            rowLst = [str(k) for k in rowLst]

            rowStr = " + ".join(rowLst) if i > 0 else str(rowLst[0])

            rowVal, rowValM = None, None


            if " + " in rowStr:
                rowVal = float(rowStr.split(" + ")[1])
            elif " - " in rowStr:
                rowValM = float(rowStr.split(" - ")[1])
            else:
                if "SoC" in rowStr:
                    rowVal = float(0)
                else:
                    rowVal = float(rowStr)

            ## FOR THE +
            if rowVal is not None:
                if prevVal is not None:
                    rowVal = prevVal + rowVal
                prevVal = rowVal

            ## FOR THE -
            if rowValM is not None:
                if prevVal is not None:
                    rowValM = prevVal - rowValM
                prevVal = rowValM

            df.at[i, colName] = rowVal if rowVal is not None else rowValM


        inptDf[colName] = df[colName]

        return inptDf



    def solveModel(self) -> pd.DataFrame:

        global windGenXlsx
        baseModel = BaseModel(fileName=windGenXlsx)

        keepSoc = [] # HOLD SOC VALUES
        
        # IMPORT THE APPROPRIATE DATA FROM THE ImportData CLASS
        importData = ImportData(**vars(baseModel))
        df = importData.windData()
        sellPrice, purchasePrice = importData.sellingPrice(), importData.purchasingPrice() # currency/MWh

        df = df["Wind"].to_frame()
        wind_profile_full = df.iloc[:, 0].tolist()


        for day in range(len(wind_profile_full) // len(self.hours)):
            wind_profile = wind_profile_full[day * len(self.hours) : (day+1) * len(self.hours)] 

            # CREATE A pulp MODEL
            model = LpProblem(name = "Battery Dispatch",
                              sense = LpMaximize)


            # ASSIGN VARIABLES
            charge = LpVariable.dicts(name = "Charge", 
                                      indices = self.hours, 
                                      lowBound = 0, 
                                      upBound = self.max_charge_rate, 
                                      cat = "Continuous")

            discharge = LpVariable.dicts("Discharge", 
                                         indices = self.hours, 
                                         lowBound = 0, 
                                         upBound = self.max_discharge_rate, 
                                         cat = "Continuous")

            total_output_poi = LpVariable.dicts("TotalOutputPOI", 
                                                indices = self.hours, 
                                                cat = "Continuous")

            charge_decision = LpVariable.dicts("Charge or not", 
                                               indices = self.hours, 
                                               cat ="Binary")

            discharge_decision = LpVariable.dicts("Discharge or not", 
                                                  indices = self.hours, 
                                                  cat = "Binary")



            soc = dict.fromkeys(list(self.hours), 0)

            ## DEFINE OBJECTIVE FUNCTION
            # ASSUMING THE SELLING PRICE REMAINS CONSTANT DURING THE HOURLY SIMULATION (300 currency/MWh)
            # IF NOT CREATE A sellPDf AND PASS THE VALUES TO THE OBJECTIVE FUNCITON FOR EVERY HOUR OF THE YEAR
            model += lpSum((total_output_poi[h] * sellPrice) for h in self.hours) # MAXIMIZE THE REVENUES OF THE HYBRID SYSTEM


            model += lpSum(charge[h] for h in self.hours) <= self.dailyCycles * self.max_daily_throughput
            model += lpSum(discharge[h] for h in self.hours) <= self.dailyCycles * self.max_daily_throughput


            for h in self.hours:

                model += charge[h] <= self.max_charge_rate # CHARGE AT HOUR h SHOULD BE LESS OR EQUAL THAN 200 MWh
                model += discharge[h] <= self.max_discharge_rate # DISCHARGE AT HOUR h SHOULD BE LESS OR EQUAL THAN 200 MWh
                model += charge_decision[h] + discharge_decision[h] <= 1 # THE BESS SHOULD ONLY CHARGE OR DISCHARGE AT HOUR h
                model += charge[h] <= self.battery_capacity - soc[h]


                ## ENERGY BALANCE
                # DEFINE INITIAL STATE OF CHARGE
                if day == 0:
                    if h == 0:
                        soc[h] = self.initialSOC
                    else:
                        soc[h] += soc[h-1]
                else:
                    if h == 0:
                        soc[h] = keepSoc[len(keepSoc) - 1] # GET THE LAST SOC VALUE FOR THE DAY
                    else:
                        soc[h] += soc[h-1]


                ## DEFINE THE PEAK HOURS
                if self.start_hour <= h <= self.end_hour:

                    ## CHARGE CONDITION
                    if wind_profile[h] > self.firm_block_limit:
                        diff = wind_profile[h] - self.firm_block_limit
                        if diff <= self.max_charge_rate:
                            if diff <= self.battery_capacity - soc[h] : chargeDiff = diff
                            else : chargeDiff = self.battery_capacity - soc[h]
                        else:
                            if self.max_charge_rate <= self.battery_capacity - soc[h] : chargeDiff = self.max_charge_rate
                            else : chargeDiff = self.battery_capacity - soc[h]

                        model += charge[h] == chargeDiff # CHARGE THE DIFFERENCE
                        model += discharge[h] == 0 # CHARGE AND DISCHARGE SHOULD NOT TAKE PLACE AT THE SAME HOUR h
                        soc[h] += chargeDiff
                        model += total_output_poi[h] == self.firm_block_limit # THE POI IS THE FIRM BLOCK LIMIT


                    ## DISCHARGE CONDITION
                    else:
                        diff = self.firm_block_limit - wind_profile[h]
                        model += charge[h] == 0 # CHARGE AND DISCHARGE SHOULD NOT TAKE PLACE AT THE SAME HOUR h

                        ## CHECK THE AMOUNT OF ENERGY STORED IN THE BESS VIA STATE OF CHARGE
                        if soc[h] >= diff:

                            if diff <= self.max_charge_rate : dischargeDiff = diff
                            else : dischargeDiff = self.max_charge_rate

                            model += discharge[h] == dischargeDiff
                            soc[h] -= dischargeDiff
                            model += total_output_poi[h] == self.firm_block_limit # THE POI IS THE FIRM BLOCK LIMIT

                        ## IF THE AMOUNT OF ENERGY STORED IN THE BESS IS NOT SUFFICIENT TO FULFILL THE FIRM BLOCK LIMIT
                        else:
                            model += discharge[h] == soc[h] # DISCHARGE THE AMOUNT OF ENERGY STORED IN THE BESS (soc[h] < diff)
                            model += total_output_poi[h] == discharge[h] + wind_profile[h] # THE POI IS THE SUM OF ENERGY GENERATED FROM WIND AND THE ENERGY STORED IN THE BESS ( < firm_block_limit)
                            soc[h] = 0 # UPDATE STATE OF CHARGE (== 0 CAUSE WE DISCHARGED ALL THE AMOUNT OF ENERGY STORED IN THE BESS)


                ## DURING THE NON-PEAK HOURS
                else:
                    model += discharge[h] == 0 # WE DO NOT NEED TO DISCHARGE ANY AMOUNT OF ENERGY DURING THE NON-PEAK HOURS

                    ## CHECK IF WE SHOULD CHARGE THE BESS
                    if wind_profile[h] > 0:

                        if wind_profile[h] <= self.max_charge_rate:
                            if wind_profile[h] <= self.battery_capacity - soc[h] : chargeAmount = wind_profile[h]
                            else : chargeAmount = self.battery_capacity - soc[h]
                        else:
                            if self.max_charge_rate <= self.battery_capacity - soc[h] : chargeAmount = self.max_charge_rate
                            else : chargeAmount = self.battery_capacity - soc[h]

                        model += charge[h] == chargeAmount # CHARGE THE BATTERY WITH THE AMOUNT OF ENERGY GENERATED FROM WIND
                        soc[h] += chargeAmount

                    else:
                        model += charge[h] == 0 # WE SHOULD NOT CHARGE THE BESS

                    model += total_output_poi[h] == wind_profile[h] - charge[h] # THE POI IS THE DIFF OF ENERGY GENERATED FROM WIND AND THE ENERGY USED TO CHARGE THE BESS


                keepSoc.append(soc[h])

            ## SOLVE THE MODEL
            model.solve()



            # print("sumCharge ", lpSum(charge[k] for k in self.hours).value())
            # print("sumDischarge ", lpSum(discharge[k] for k in self.hours).value())

            # for h in self.hours:
            #     print("h {}  charge {}   discharge {}   soc {}   wind {}".format(h,
            #                                                                      charge[h].value(),
            #                                                                      discharge[h].value(),
            #                                                                      [soc[h].value() if (type(soc[h]) != float and type(soc[h]) != int) else soc[h]][0],
            #                                                                      wind_profile[h]))


            for hour in self.hours:
                self.results.append(
                    {
                        "Day" : day,
                        "Hour" : hour,
                        "Charge(MWh)" : charge[hour].value(),
                        "Discharge(MWh)" : discharge[hour].value(),
                        "SoC(MWh)" : soc[hour],
                        "Wind to POI(MWh)" : wind_profile[hour],
                        "W-F" : wind_profile[hour] - self.firm_block_limit,
                        "isPeakHour" : self.start_hour <= hour <= self.end_hour,
                        "Total Output at POI(MWh)" : total_output_poi[hour].value(),
                        "Revenues(currency)" : total_output_poi[hour].value() * sellPrice
                    }
                )


        resultFileName = r"BESSOptResults.xlsx"
        optimalDf = pd.DataFrame(self.results)

        ## CHANGE THE STATE OF CHARGE COLUMN OF THE DATAFRAME 
        # optimalDf = self.adjustCol(inptDf = optimalDf)
        optimalDf["Datetime"] = pd.to_datetime(optimalDf["Hour"], unit = "h", origin = "2022-01-01")
        optimalDf = optimalDf.set_index("Datetime")

        # optimalDf['CumulativeRevenues(currency)'] = optimalDf.groupby('Day')['Revenues(currency)'].cumsum()

        optimalDf.to_excel(self.searchPath + resultFileName, index=False)

        return optimalDf



if __name__ == "__main__":

    baseModel = BaseModel(fileName=r"gpwind2021gross.xlsx")
    opt = OptimizeModel(**vars(baseModel))

    ansDf = opt.solveModel()

    totalCharge, totalDischarge = ansDf["Charge(MWh)"].sum(), ansDf['Discharge(MWh)'].sum()
    totalWind = ansDf["Wind to POI(MWh)"].sum()
    totalYield, totalRevs = ansDf["Total Output at POI(MWh)"].sum(), ansDf['Revenues(currency)'].sum()

    print("Yearly Energy Charge = {} (MWh)\nYearly Energy Discharge = {} (MWh)".format(format(totalCharge, ","),
                                                                                       format(totalDischarge, ",")))
    print("Yearly Energy Wind = {} (MWh)".format(format(totalWind, ",")))

    print("Yearly Hybrid Energy Yield = {} (MWh)\nYearly Hybrid Total Revenues = {} (currency)".format(format(totalYield, ","),
                                                                                                       format(totalRevs, ",")))

