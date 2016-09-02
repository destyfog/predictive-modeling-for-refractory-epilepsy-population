-- events2 = LOAD 's3://8803bdh/event2.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, timestamp:chararray, value:float); 
events2 = LOAD 's3://8803bdh/events3.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, timestamp:chararray, value:float); 
----delete header
events2 = filter events2 by ($0 is not null);


--events2 = LOAD 'data/event.csv' USING PigStorage(',') AS (patientid:int, eventid:chararray, timestamp:chararray, value:float); 


events = FOREACH events2 GENERATE patientid, eventid, ToDate(timestamp, 'yyyy-MM-dd') AS etimestamp, value;

epilepsy_dx_hash = LOAD 's3://8803bdh/epilepsy_dx_hash.csv' USING PigStorage(',') AS (realevent:chararray,eventid:chararray); 

-- epilepsy_dx_hash = LOAD 'data/epilepsy_dx_hash.csv' USING PigStorage(',') AS (realevent:chararray,eventid:chararray);  

diaevents = FILTER events BY (eventid matches 'DIAG.*');

-- dxtmp = cross diaevents , epilepsy_dx_hash;
dxtmp = join diaevents by eventid , epilepsy_dx_hash by eventid;

--dxtmp2 = filter dxtmp by (diaevents::eventid == epilepsy_dx_hash::eventid);

dx_780 = filter dxtmp by (diaevents::eventid == 'DIAG_14e455c8e4a5b1b1e0b1062048188abe4bc5d059');

dx_780_2 = FOREACH dx_780 GENERATE diaevents::patientid as patientid;
dx_780grouped = group dx_780_2 by patientid;

X = FOREACH dx_780grouped GENERATE group as patientid, COUNT(dx_780_2) as count;
------------
dx_780id = foreach (filter X by (count >=(long)2)) GENERATE patientid as patientid;
------------

dx_non780 = filter dxtmp by (diaevents::eventid != 'DIAG_14e455c8e4a5b1b1e0b1062048188abe4bc5d059');

dx_non780_2 = FOREACH dx_non780 GENERATE diaevents::patientid as patientid;

---------
dx_non780id = DISTINCT dx_non780_2;
---------
dx = union dx_non780id, dx_780id;
dx = DISTINCT dx;

moleculeevents = FILTER events BY (eventid matches 'MOLECULE.*');

aedlist = LOAD 's3://8803bdh/aed_list.txt' USING PigStorage(',') AS (aed:chararray); 

-- aedlist = LOAD 'data/aed_list.txt' USING PigStorage(',') AS (aed:chararray); 

rxtmpfilter = join moleculeevents by eventid , aedlist by aed;

--rxtmpfilter = filter rxtmp by (moleculeevents::eventid == aedlist::aed);

rxtmp2 = foreach rxtmpfilter GENERATE moleculeevents::patientid as patientid, moleculeevents::eventid as aed, moleculeevents::etimestamp as etimestamp;

rxgrouped = group rxtmp2 by patientid;

rxgrouped2 = foreach rxgrouped GENERATE group as patientid, MIN(rxtmp2.etimestamp) as earliestdate;
--this is index date----------

yobevents = FILTER events BY (eventid matches 'YOB.*');

join_rx_yob = JOIN yobevents by $0, rxgrouped2 by $0;


findage = foreach join_rx_yob GENERATE yobevents::patientid as rxpatientid , YearsBetween(rxgrouped2::earliestdate,yobevents::etimestamp) as ageatfirstaed;

agefilter = filter findage by (ageatfirstaed>=(long)16);

rx_age_id = foreach agefilter GENERATE rxpatientid as patientid;

general_population = join rx_age_id by $0, dx by $0;

general_population_id = foreach general_population GENERATE rx_age_id::patientid as patientid;

general_population_id_idx = join general_population_id by $0, rxgrouped2 by $0;

general_population_id_idx = foreach general_population_id_idx GENERATE general_population_id::patientid as patientid, rxgrouped2::earliestdate as indexdate;

--store general_population_id into 's3://8803bdh/general_population_id';

--store rxgrouped2 into 's3://8803bdh/patientid_indexdate_first_aed';

aedcount = FOREACH rxgrouped GENERATE group as patientid, COUNT(rxtmp2) as count;

general_population_id_idx_aedcount = join general_population_id_idx by $0, aedcount by $0;

general_population_id_idx_aedcount = foreach general_population_id_idx_aedcount GENERATE general_population_id_idx::patientid as patientid, general_population_id_idx::indexdate as indexdate, aedcount::count as aedcount;

general_population_id_idx_aedcount = DISTINCT general_population_id_idx_aedcount;

casepatient = filter general_population_id_idx_aedcount by (aedcount>=(long)5);
casepatient = foreach casepatient GENERATE *, (int)1 as label;

controlpatient = filter general_population_id_idx_aedcount by (aedcount==(long)1);
controlpatient = foreach controlpatient GENERATE *, (int)0 as label;

allcasecontrol = union casepatient, controlpatient;

allcasecontrol = foreach allcasecontrol GENERATE patientid as patientid, indexdate as indexdate, label as label;

allccevent = join allcasecontrol by $0, events by $0;

allccevent = foreach allccevent GENERATE allcasecontrol::patientid as patientid, allcasecontrol::indexdate as indexdate, allcasecontrol::label as label, events::eventid as eventid, events::etimestamp as eventdate, events::value as value;
allccevent = foreach allccevent GENERATE *, DaysBetween(indexdate,eventdate) as time_difference;

--filter data after index date
allfilter = filter allccevent by (time_difference >=(long)0 and time_difference <= (long)365 and value is not null);

allfilter_2 = foreach allfilter GENERATE patientid, label, eventid,value;

-- aggregate

----split count and sum
-----(patientid, label, eventid,value)
needsum = filter allfilter_2 by (eventid matches '(GENDER|HOSPDAYS|SUPPLYDAYS|QUANTITY).*') ;
needsum_grouped = group needsum by (patientid,label,eventid);

needsum_features = -- do sum
FOREACH needsum_grouped GENERATE group.patientid as patientid,group.label as label, group.eventid as eventid, SUM(needsum.value) as featurevalue;
-----?

needcount = filter allfilter_2 by (eventid matches '(INPATIENT|OUTPATIENT|ER|DIAG|PROC|MOLECULE|CLASS).*') ;

needcount_grouped = group needcount by (patientid,label,eventid);

needcount_features = -- for group of (patientid, eventid), count the number of  events occurred for the patient and create it (patientid, eventid, featurevalue)
FOREACH needcount_grouped GENERATE group.patientid as patientid,group.label as label, group.eventid as eventid, COUNT(needcount) as featurevalue;

-- rank events
----UNION sum and count
all_features = union needsum_features, needcount_features;

grouped = group all_features by eventid;
eventidtable = FOREACH grouped GENERATE group;
groupeddistinct = DISTINCT eventidtable;
g_order = ORDER groupeddistinct by group ASC;

featuresmap =rank g_order by group;

featuresmap = FOREACH featuresmap GENERATE ($0-(long)1) as idx, $1 as eventid;

joinwithmap = join featuresmap by $1, all_features by eventid;

features = foreach joinwithmap GENERATE all_features::patientid as patientid ,all_features::label as label,featuresmap::idx as idx, all_features::featurevalue as value;

-- Normalize

maxvalues = -- group events by idx and compute the maximum feature value in each group, is of the form (idx, maxvalues)
FOREACH (group features by idx) GENERATE group as idx, MAX(features.value) as maxvalue;

normalized = -- join features and maxvalues by idx
JOIN features by $2 LEFT OUTER, maxvalues BY $0;

normalized_rename = FOREACH normalized GENERATE features::patientid as patientid,features::label as label, features::idx as idx,features::value as featurevalue,maxvalues::maxvalue as maxvalues;

n1 = filter normalized_rename by maxvalues==0;
n2 = filter normalized_rename by maxvalues!=0;

-- featuresnormalized = -- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)
-- FOREACH normalized_rename GENERATE patientid,label, idx, ((DOUBLE)featurevalue / (DOUBLE)maxvalues) as normalizedfeaturevalue;
------------------------------------------
feature1 = foreach n1 GENERATE patientid, idx, (DOUBLE)featurevalue as normalizedfeaturevalue;

features2 = -- compute the final set of normalized features of the form (patientid, idx, normalizedfeaturevalue)
FOREACH n2 GENERATE patientid, idx, ((DOUBLE)featurevalue / (DOUBLE)maxvalues) as normalizedfeaturevalue;
features = union feature1, features2;
-----?
-- Generate features in svmlight format

REGISTER 's3://8803bdh/utils.py' USING jython AS utils;

--grpd = GROUP features BY (patientid,label);
grpd = GROUP features BY patientid;

grpd_order = ORDER grpd BY $0;

features = FOREACH grpd_order 
{
    sorted = ORDER features BY idx;
    generate group as patientid, utils.bag_to_svmlight(sorted) as sparsefeature;
}

-- samples = FOREACH grpd_order 
-- {
--     sorted = ORDER features BY idx;
--     generate group, utils.bag_to_svmlight(sorted) as sparsefeature;
-- }

-- Split into train and test set

labels = foreach normalized_rename GENERATE patientid, label;
labels = DISTINCT labels;

----Generate sparsefeature vector relation
samples = JOIN features BY patientid, labels BY patientid;
samples = DISTINCT samples PARALLEL 1;
samples = ORDER samples BY $0;
samples = FOREACH samples GENERATE $3 AS label, $1 AS sparsefeature;


---- randomly split data for training and testing
samples = FOREACH samples GENERATE RANDOM() as assignmentkey, *;
SPLIT samples INTO testing IF assignmentkey <= 0.20, training OTHERWISE;
training = FOREACH training GENERATE $1..;
testing = FOREACH testing GENERATE $1..;

---- save training and tesing data
STORE testing INTO 's3://8803bdh/superbig/2/testing' USING PigStorage(' ');
STORE training INTO 's3://8803bdh/superbig/2/training' USING PigStorage(' ');

