-- Selecting all columns from the heart_disease_risk table for all records
SELECT *
FROM heart_disease_risk;

-- Selecting specific columns from the heart_disease_risk table for all records
SELECT age, sex, cp, trestbps, chol, thalach, decision
FROM heart_disease_risk;

-- Selecting all columns from the heart_disease_risk table for records with a specific diagnosis
SELECT *
FROM heart_disease_risk
WHERE decision = 1;

-- Selecting records with specific conditions based on multiple columns
SELECT *
FROM heart_disease_risk
WHERE age > 50 AND sex = 1 AND cp = 2;

-- Selecting records sorted in ascending order by age
SELECT *
FROM heart_disease_risk
ORDER BY age;

-- Selecting records sorted in descending order by cholesterol level and age
SELECT *
FROM heart_disease_risk
ORDER BY chol DESC, age DESC;

-- Selecting the average age, maximum cholesterol level, and minimum resting blood pressure
SELECT AVG(age) AS average_age, MAX(chol) AS max_cholesterol, MIN(trestbps) AS min_resting_bp
FROM heart_disease_risk;

-- Using the DISTINCT keyword to select unique values of the chest pain type
SELECT DISTINCT cp
FROM heart_disease_risk;

-- Using the LIKE operator to select records with a specific pattern in the file_name column
SELECT *
FROM heart_disease_risk
WHERE file_name LIKE '%data_record%.csv';

-- Using aggregate functions to calculate the total number of male and female patients
SELECT sex, COUNT(*) AS total_count
FROM heart_disease_risk
GROUP BY sex;

-- Using the LIMIT clause to retrieve a specific number of records
SELECT *
FROM heart_disease_risk
LIMIT 10;
