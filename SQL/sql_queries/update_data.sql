-- Updating data in the heart_disease_risk table
UPDATE heart_disease_risk
SET chol = 240.0
WHERE id = 1;

-- Updating the diagnosis for a specific record based on the ID
UPDATE heart_disease_risk
SET decision = 1
WHERE id = 1;

-- Incrementing the age of all male patients by 1
UPDATE heart_disease_risk
SET age = age + 1
WHERE sex = 1;

-- Updating multiple columns for records with specific conditions
UPDATE heart_disease_risk
SET chol = chol - 10, thalach = thalach + 5
WHERE age > 60 AND decision = 1;

-- Using the CASE statement to update a column based on different conditions
UPDATE heart_disease_risk
SET decision = CASE
                  WHEN age <= 50 THEN 0
                  WHEN age > 50 AND age <= 70 THEN 1
                  ELSE 2
              END;

-- Updating a column based on a subquery
UPDATE heart_disease_risk
SET decision = 1
WHERE id IN (
    SELECT id
    FROM other_table
    WHERE condition = 'some_value'
);

-- Updating records using values from another table with a JOIN
UPDATE heart_disease_risk AS h
SET chol = o.new_chol
FROM other_table AS o
WHERE h.id = o.id;

-- Updating records with random values for specific columns
UPDATE heart_disease_risk
SET chol = FLOOR(RANDOM() * 100) + 150, thalach = FLOOR(RANDOM() * 30) + 150
WHERE decision = 1;
