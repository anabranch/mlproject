SELECT a.pipeline_number, a.metric, a.value, b.metric, b.value
     FROM metrics a
     INNER JOIN metrics b
     ON (a.pipeline_number = b.pipeline_number)
     WHERE a.metric='validation score'
           AND (b.metric ='best_estimator' or b.metric='XY13')
     ORDER BY cast(a.value as real);
