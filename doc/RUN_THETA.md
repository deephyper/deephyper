# RUN ON THETA

```
qsub -n 1 -t 20 -q debug-cache-quad -I -A datascience
```

```
nodelist | grep debug | grep idle
```
