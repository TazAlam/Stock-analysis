
from pyspark.sql import SparkSession
from pyspark.sql.functions import date_add, to_date, format_number, year, quarter,month
from pyspark.sql.types import DateType

import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import seaborn as sns

from scipy.stats.stats import spearmanr
from scipy.stats.stats import kendalltau

spark = SparkSession \
    .builder \
    .appName("Python Spark create RDD example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


test = spark.read.csv("/Users/tahmidalam/Documents/University/Big Data Analytics/Assignment 2/amex-nyse-nasdaq-stock-histories1/history_60d.csv", header=True)


# date is currently in yyyy - mm - dd format and saved as strings.

test = test.withColumn("date", test["date"].cast(DateType()))

#date now in the correct format

test = test.withColumn("Month",month(test['date']))
test = test.withColumn("Year", year(test['date']))
test = test.withColumn("Quarter", quarter(test['date']))

test.show(10)
# date now in the correct format, and month,year and quarter columns now created

# following Nasdaq stock:

#  IT stock:  AAPL = apple, FB = Facebook, ADBE = adobe, AKAM = akamai technologies, ULTI = ultimate softwate, MGIC = magic software entp
#  healthcare: ELOX = eloxx pharma, ENOB = enochian biosciences, NXGN = nextgen healthcare, SRTS = sensus healthcare, TRHC = Tabula Rasa healthcare
# Finance: , WHF = whitehorse finance, WSBF = waterstone financial, AFH = Atlas finc, CMFN = Cm finance, PTMN = portman ridge finance corp


copydf = test

# copied just for replication purposes

#----------------------------------------------
#IT DF manipulation

filterA = copydf.symbol == 'AAPL'
filterB = copydf.symbol == 'FB'
filterC = copydf.symbol == 'ADBE'
filterD = copydf.symbol == 'AKAM'
filterE = copydf.symbol == 'ULTI'

ITdf = copydf.filter(filterA | filterB | filterC | filterD | filterE)

#Df now reduced to just contain the stock of importance

# now create a column calculating the change of valuation throughout the day

ITdf = ITdf.withColumn("total-variation", ITdf.high - ITdf.low)                  #variation column now created

# calculate the average for each group individually

AAPL_avg_variation = ITdf.filter(ITdf['symbol'] == 'AAPL').agg({"total-variation": "avg"})
FB_avg_variation = ITdf.filter(ITdf['symbol'] == 'FB').agg({"total-variation": "avg"})
ADBE_avg_variation = ITdf.filter(ITdf['symbol'] == 'ADBE').agg({"total-variation": "avg"})
AKAM_avg_variation = ITdf.filter(ITdf['symbol'] == 'AKAM').agg({"total-variation": "avg"})
ULTI_avg_variation= ITdf.filter(ITdf['symbol'] == 'ULTI').agg({"total-variation": "avg"})



print("FB average variation is:")
FB_avg_variation.show()
# 3.0195243472144724

print("AAPL average varation is:")
AAPL_avg_variation.show()
# 3.1033346993582596

print("ADBE average variation is:")
ADBE_avg_variation.show()
#  4.914284115745898

print("AKAM average variation is:")
AKAM_avg_variation.show()
# 1.1847619556245346

print("ULTI average variation is:")
ULTI_avg_variation.show()

# 0.8747638520740327

ITVarSum = 3.0195243472144724 + 3.1033346993582596 + 4.914284115745898 +\
           1.1847619556245346 + 0.8747638520740327

Average_Variation_IT = ITVarSum / 5

#Average variation of IT stock now calculated,printed 2.619333794003439

#------------------------------------------------------------------

# create the Healthcare DF
#rename testdf to original Df:
originalDF = test
#apply filters for the stock:  ELOX, ENOB, NXGN, SRTS, TRHC
filterF = originalDF.symbol == 'ELOX'
filterG = originalDF.symbol == 'ENOB'
filterH = originalDF.symbol == 'NXGN'
filterI = originalDF.symbol == 'SRTS'
filterJ = originalDF.symbol == 'TRHC'

HealthcareDF = originalDF.filter(filterF | filterG | filterH | filterI | filterJ)

HealthcareDF = HealthcareDF.withColumn("total-variation", HealthcareDF.high - HealthcareDF.low)

ELOX_avg_variation = HealthcareDF.filter(HealthcareDF['symbol'] == 'ELOX').agg({"total-variation": "avg"})
ENOB_avg_variation = HealthcareDF.filter(HealthcareDF['symbol'] == 'ENOB').agg({"total-variation": "avg"})
NXGN_avg_variation = HealthcareDF.filter(HealthcareDF['symbol'] == 'NXGN').agg({"total-variation": "avg"})
SRTS_avg_variation = HealthcareDF.filter(HealthcareDF['symbol'] == 'SRTS').agg({"total-variation": "avg"})
TRHC_avg_variation = HealthcareDF.filter(HealthcareDF['symbol'] == 'TRHC').agg({"total-variation": "avg"})

#print("ELOX average variation is:")
#ELOX_avg_variation.show() #0.8854761123657227
#print("ENOB average variation is:")
#ENOB_avg_variation.show() #0.36473803293137314
#print("NXGN average variation is:")
#NXGN_avg_variation.show() #0.4383335113525392
#print("SRTS average variation is:")
#SRTS_avg_variation.show() #0.3160237584795271
#print("TRHC average variation is:")
#TRHC_avg_variation.show() #3.048523857480003

Average_variation_healthcare = (0.8854761123657227 +  0.36473803293137314 + 0.4383335113525392 +
                                0.3160237584795271 + 3.048523857480003) /5
print("Average variation in Healthcare:", Average_variation_healthcare) #1.010619054521833

#---------------------------
# Finance DF
#first the filters for: WHF, WSBF, AFH, CMFN, PTMN
filterK = originalDF.symbol == 'WHF'
filterL = originalDF.symbol == 'WSBF'
filterM = originalDF.symbol == 'AFH'
filterN = originalDF.symbol == 'CMFN'
filterO = originalDF.symbol == 'PTMN'

FinanceDF = originalDF.filter(filterK | filterL | filterM | filterN | filterO)

FinanceDF = FinanceDF.withColumn("total-variation", FinanceDF.high - FinanceDF.low)

WHF_avg_variation = FinanceDF.filter(FinanceDF['symbol'] == 'WHF').agg({"total-variation": "avg"})
WSBF_avg_variation = FinanceDF.filter(FinanceDF['symbol'] == 'WSBF').agg({"total-variation": "avg"})
AFH_avg_variation = FinanceDF.filter(FinanceDF['symbol'] == 'AFH').agg({"total-variation": "avg"})
CMFN_avg_variation = FinanceDF.filter(FinanceDF['symbol'] == 'CMFN').agg({"total-variation": "avg"})
PTMN_avg_variation = FinanceDF.filter(FinanceDF['symbol'] == 'PTMN').agg({"total-variation": "avg"})

#print("WHF avg variation:")
#WHF_avg_variation.show()       #  0.26476199286324636
#print("WSBF avg variation:")
#WSBF_avg_variation.show()       # 0.22500013169788197
#print("AFH avg variation:")
#AFH_avg_variation.show()        # 0.2961905286425638
#print("CMFN avg variation:")
#CMFN_avg_variation.show()       #  0.168428557259696
#print("PTMN avg variation:")
#PTMN_avg_variation.show()       # 0.10159998893737802

Average_variation_finance = (0.26476199286324636 + 0.22500013169788197 + 0.2961905286425638 +
                             0.10159998893737802 ) /5

print( "Average finance variation: ", Average_variation_finance)
data = [['IT', Average_Variation_IT], ['Healthcare', Average_variation_healthcare], ['Finance', Average_variation_finance]]

AverageVariationDF = pd.DataFrame(data, columns = ['Field', 'Average Variation per day'])
#pd df now created with all average variation for the stock fields

print(AverageVariationDF)

#---------------------------------------------------------------------------
#now attempt to plot some of this data in regards to the date of trade.
#i.e, see how the variation changes with time, also potentially calculate average variation per year(or quarter) and also plot.


# create  visualisation of:
#       1 time series for each DF, where each respective stock is plotted
#       a time series comparing average variation between stock
#       time series comparing quarterly averages against each other
#               - may need to calculate average variation per quarter for each stock, and then for the DF
#       look into k-means clustering
#       Calculate average variation PER YEAR for each main DF, then amalgamate into one DF
#       from this, plot as multiple time series

# create a a pandas df to allow for visualisation first

FinancePDF = FinanceDF.toPandas()
ITPDF = ITdf.toPandas()
HealthcarePDF = HealthcareDF.toPandas()

# remember to change to 'datetime' using : df['date_column'] = pd.to_datetime(df['date_column'])

FinancePDF['date'] = pd.to_datetime(FinancePDF['date'])
ITPDF['date'] = pd.to_datetime(ITPDF['date'])
HealthcarePDF['date'] = pd.to_datetime(HealthcarePDF['date'])

#create subdf's for each stock, then join each of them side by side to create new DF for the analysis
#  IT stock:  AAPL = apple, FB = Facebook, ADBE = adobe, AKAM = akamai technologies, ULTI = ultimate softwate, MGIC = magic software entp

#-------------------------------------------------------------------------------------------
# Creation of Trimmed IT Df creation
filterAAPL = ITPDF['symbol'] == 'AAPL'
filterFB = ITPDF['symbol'] == 'FB'                                                      # creating filters
filterADBE = ITPDF['symbol'] == 'ADBE'
filterAKAM = ITPDF['symbol'] == 'AKAM'
filterULTI = ITPDF['symbol'] == 'ULTI'

appleDF = ITPDF[filterAAPL]
facebookDF = ITPDF[filterFB]                                                            #appling filters to create sole DF's
adobeDF = ITPDF[filterADBE]
akamaiDF = ITPDF[filterAKAM]
ultiDF = ITPDF[filterULTI]

appleSubDF = appleDF[['date', 'total-variation']]
facebookSubDF = facebookDF[['date', 'total-variation']]
adobeSubDF = adobeDF[['date', 'total-variation']]                                           #new df's subsetting to include only date & total variation
akamaiSubDF = akamaiDF[['date', 'total-variation']]
ultiSubDF = ultiDF[['date', 'total-variation']]


testDF = appleSubDF.merge(facebookSubDF, on ='date')                                    # merge all sub df's into one df's, each column
testDF2 = testDF.merge(adobeSubDF, on = 'date')
testDF3 = testDF2.merge(akamaiSubDF, on = 'date')
testDF4 = testDF3.merge(ultiSubDF, on = 'date')

ITtrimmedDF = testDF4                                                                #trimmed IT DF now created for analysis, columns (left to right) go:
                                                                                         # apple, facebook, adobe, akamai, ulti

ITtrimmedDF.columns = ['date', 'AAPL Variation', 'FB Variation',
                       'ADBE Variation', 'AKAM Variation', 'ULTI Variation']

print(ITtrimmedDF.head(3))

# columns now renamed successfully.

#attempt rolling average timeseries
# ma = co2_levels.rolling(window =52).mean()

              #successful but still a little unclear, maybe try rolling averages??

#----------------------------------------------------------------------------------------------
# Healthcare
#  healthcare: ELOX = eloxx pharma, ENOB = enochian biosciences, NXGN = nextgen healthcare, SRTS = sensus healthcare, TRHC = Tabula Rasa healthcare

filterELOX = HealthcarePDF['symbol'] == 'ELOX'
filterENOB = HealthcarePDF['symbol'] == 'ENOB'
filterNXGN = HealthcarePDF['symbol'] == 'NXGN'
filterSRTS = HealthcarePDF['symbol'] == 'SRTS'
filterTRHC = HealthcarePDF['symbol'] == 'TRHC'

eloxDF = HealthcarePDF[filterELOX]
enochianDF = HealthcarePDF[filterENOB]
nextgenDF = HealthcarePDF[filterNXGN]
senusDF = HealthcarePDF[filterSRTS]
tabulaDF = HealthcarePDF[filterTRHC]


eloxSubDF = eloxDF[['date', 'total-variation']]
enochianSubDF = enochianDF[['date', 'total-variation']]
nextgenSubDF = nextgenDF[['date', 'total-variation']]
sensusSubDF = senusDF[['date', 'total-variation']]
tabulaSubDF = tabulaDF[['date', 'total-variation']]

mergeDF = eloxSubDF.merge(enochianSubDF, on = 'date')
mergeDF2 = mergeDF.merge(nextgenSubDF, on = 'date')
mergeDF3 = mergeDF2.merge(sensusSubDF, on = 'date')
mergeDF4 = mergeDF3.merge(tabulaSubDF, on = 'date')

HealthcareTrimmedDF = mergeDF4

HealthcareTrimmedDF.columns = ['date', 'ELOX Variation', 'ENOB Variation',
                               'NXGN Variation', 'SRTS Variation', 'TRHC Variation']

#trimmed healthcare DF created  successfully.


#------------------------------------
#Finance
# Finance: , WHF = whitehorse finance, WSBF = waterstone financial, AFH = Atlas finc, CMFN = Cm finance, PTMN = portman ridge finance corp

WHFfilter = FinancePDF['symbol'] == 'WHF'
WSBFfilter = FinancePDF['symbol'] == 'WSBF'
AFHfilter = FinancePDF['symbol'] == 'AFH'
CMFNfilter = FinancePDF['symbol'] == 'CMFN'
PTMNfilter = FinancePDF['symbol'] == 'PTMN'

whitehorseDF = FinancePDF[WHFfilter]
waterstoneDF = FinancePDF[WSBFfilter]
atlasDF = FinancePDF[AFHfilter]
cmDF = FinancePDF[CMFNfilter]
portmanDF = FinancePDF[PTMNfilter]

whitehorseSubDF = whitehorseDF[['date', 'total-variation']]
waterstoneSubDF = waterstoneDF[['date', 'total-variation']]
atlasSubDF = atlasDF[['date', 'total-variation']]
cmSubDF = cmDF[['date', 'total-variation']]
portmanSubDF = portmanDF[['date', 'total-variation']]

joindf = whitehorseSubDF.merge(waterstoneSubDF, on = 'date')
joindf2 = joindf.merge(atlasSubDF, on = 'date')
joindf3 = joindf2.merge(cmSubDF, on = 'date')
joindf4 = joindf3.merge(portmanSubDF,how = 'left', on = 'date')


FinanceTrimmedDF = joindf4

FinanceTrimmedDF.columns = ['date', 'WHF Variation', 'WSBF Variation', 'AFH Variation',
                            'CMFN Variation', 'PTMN Variation']

#trimmed finance df created successfully.

#plt.show()                 #graph shows the the variation for each

#---------------------------------------------------------
          #42 observations in ITtrimmedDf

          #42 observations in HealthcareTrimmedDF

          #25 observations in FinanceTrimmedDF

print("Observations for the sub df's of Apple:", appleSubDF.shape[0], "facebook:", facebookSubDF.shape[0], "adobe:", adobeSubDF.shape[0],
      "akamai:", akamaiSubDF.shape[0], "ultimate:", ultiSubDF.shape[0])

print("Observations for the DF's of Elox:", eloxDF.shape[0], "enochian: ", enochianDF.shape[0], "next gen: ", nextgenDF.shape[0], "senus: ",
      senusDF.shape[0], "tabula: ", tabulaDF.shape[0])

print("Observations for the finance DF's, whitehorse " , whitehorseDF.shape[0], "waterstone:", waterstoneDF.shape[0], "atlas:", atlasDF.shape[0],
      "cm:", cmDF.shape[0], "portman:", portmanDF.shape[0])


#all stock in Healthcare have equal number of observations
# all the stock in IT have equal number of observations
#portman has only 25 observation! therefore this is then trims the whole finance DF down to 25.

#print(ITtrimmedDF.head(10))         #2019-04-18         this is the final date
#print(ITtrimmedDF.tail(20))         #2019-02-20         this is the first date!



#----------------------------------------------------------------------------------------------------------
#create a df holding the 'average' variation of each stock, per day, then plot
#create a new column in the subDF's 'average IT variation' etc, and then subset this into its separate df
# merge all these df's together, so the new df contains the average variation of each day, for each field




#creating the average variation columns first:

ITtrimmedDF['Average IT Variation'] = (ITtrimmedDF['AAPL Variation'] + ITtrimmedDF['FB Variation'] + ITtrimmedDF['ADBE Variation'] +
                                       ITtrimmedDF['AKAM Variation'] + ITtrimmedDF['ULTI Variation']) /5

HealthcareTrimmedDF['Average Healthcare Variation'] = (HealthcareTrimmedDF['ELOX Variation'] + HealthcareTrimmedDF['ENOB Variation'] +
                                                       HealthcareTrimmedDF['NXGN Variation'] + HealthcareTrimmedDF['SRTS Variation'] +
                                                       HealthcareTrimmedDF['TRHC Variation']) /5

FinanceTrimmedDF['Average Finance Variation'] = (FinanceTrimmedDF['WHF Variation'] + FinanceTrimmedDF['WSBF Variation']
                                                 + FinanceTrimmedDF['AFH Variation'] + FinanceTrimmedDF['CMFN Variation']
                                                 + FinanceTrimmedDF['PTMN Variation']) /5

#average columns now created

#now subset these down so its just the date and the average column present.

AverageITVariationDF = ITtrimmedDF[['date', 'Average IT Variation']]
AverageHealthcareVariationDF = HealthcareTrimmedDF[['date', 'Average Healthcare Variation']]
AverageFinanceVariationDF = FinanceTrimmedDF[['date', 'Average Finance Variation']]

#all df's above are of same size, merge as normal.

temp = AverageITVariationDF.merge(AverageHealthcareVariationDF, on = 'date')
temp2 = temp.merge(AverageFinanceVariationDF, on ='date')
AvgVarDF = temp2

print(type(AvgVarDF.head(3)))

#df created with the average variation for each field, through time!


#"rows of IT is:

#----------------------------------------------------------------------------------------------------------

# Analysis

ITtrimmedDF = ITtrimmedDF.set_index('date')

ax =  ITtrimmedDF.plot(linewidth = 1)

ax.set_xlabel('Date', fontsize = 8)
ax.set_ylabel('Variation', fontsize = 8)                        #graph showing how individual IT stock varies
ax.set_title('IT stock variation through time')
ax.legend(fontsize = 8)



HealthcareTrimmedDF = HealthcareTrimmedDF.set_index('date')
ax =  HealthcareTrimmedDF.plot(linewidth = 1)

ax.set_xlabel('Date', fontsize = 8)
ax.set_ylabel('Variation', fontsize = 8)                        #graph showing how individual Healthcare stock varies
ax.set_title('Healthcare stock variation through time')
ax.legend(fontsize = 8)



FinanceTrimmedDF = FinanceTrimmedDF.set_index('date')
ax =  FinanceTrimmedDF.plot(linewidth = 1)

ax.set_xlabel('Date', fontsize = 8)                             #graph showing how individual Finance stock varies
ax.set_ylabel('Variation', fontsize = 8)
ax.set_title('Finance stock variation through time')
ax.legend(fontsize = 8)



AverageVariationDF.plot(kind = 'bar', x = 'Field', y = 'Average Variation per day', fontsize = 6)

#plt.show()                                                      #bar chart that visualises the comparison of average variation between the different fields



AvgVarDF = AvgVarDF.set_index('date')
ax = AvgVarDF.plot(linewidth = 1 , fontsize =6)
Avg_summary = AvgVarDF.describe()

ax.table(cellText = Avg_summary.values,
         colWidths = [0.15] *len(AvgVarDF.columns),
         rowLabels = Avg_summary.index,
         colLabels = Avg_summary.columns,
         loc = 'top')

#how do i get the table to stop cutting off???

ax.set_xlabel('Date', fontsize = 8)
ax.set_ylabel('Variation', fontsize = 8)                            #graph to visualise the average variation in value, between different stock types
#ax.set_title('')

ax.legend(fontsize = 6)

#plt.show()

#--------------------
#correlation Testing
print(ITtrimmedDF.head(10))

correlation_It_p = ITtrimmedDF[['AAPL Variation','FB Variation', 'ADBE Variation', 'AKAM Variation', 'ULTI Variation']].corr(method = 'pearson')
correlation_It_s = ITtrimmedDF[['AAPL Variation','FB Variation', 'ADBE Variation', 'AKAM Variation', 'ULTI Variation']].corr(method = 'spearman')

correlation_Healthcare_p = HealthcareTrimmedDF[['ELOX Variation', 'ENOB Variation', 'NXGN Variation', 'SRTS Variation', 'TRHC Variation']].corr(method = 'pearson')
correlation_Healthcare_s = HealthcareTrimmedDF[['ELOX Variation', 'ENOB Variation', 'NXGN Variation', 'SRTS Variation', 'TRHC Variation']].corr(method = 'spearman')

correlation_Finance_p = FinanceTrimmedDF[['WHF Variation', 'WSBF Variation', 'AFH Variation', 'CMFN Variation', 'PTMN Variation']].corr(method = 'pearson')
correlation_Finance_s = FinanceTrimmedDF[['WHF Variation', 'WSBF Variation', 'AFH Variation', 'CMFN Variation', 'PTMN Variation']].corr(method = 'spearman')


correlation_all_p = AvgVarDF[['Average IT Variation', 'Average Healthcare Variation', 'Average Finance Variation']].corr(method = 'pearson')

print(correlation_all_p)

#possible colours:
#'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
# 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r'

sns.clustermap(correlation_It_p,
               annot = True,
               linewidths = 0.4,
               annot_kws = {"size": 10},
               figsize = (5,5),
               cmap = "Blues")

sns.clustermap(correlation_Healthcare_p,
               annot = True,
               linewidths = 0.4,
               annot_kws = {"size": 10},
               figsize = (5,5),
               cmap = "Greens")


sns.clustermap(correlation_Finance_p,
               annot = True,
               linewidths = 0.4,
               annot_kws = {"size": 10},
               figsize = (5,5),
               cmap = "Reds")

sns.clustermap(correlation_all_p,
               annot = True,
               linewidths = 0.4,
               annot_kws = {"size": 10},
               figsize = (5,5),
               cmap = "Reds")


plt.show()
