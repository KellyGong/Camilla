
###
 # @Author: your name
 # @Date: 2022-01-23 11:02:46
 # @LastEditTime: 2022-02-25 11:22:22
 # @LastEditors: Please set LastEditors
 # @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 # @FilePath: /regression-cd/multiple_run.sh
### 
dataset=titanic
cdm=Camilla


for runseed in 1 21 42 84 168 336 672 1344 2688 5376
do
nohup python cognitive_diagnose_seed.py --cdm $cdm --dataset $dataset --seed $runseed > "logs/log_${cdm}_${dataset}_${runseed}" &
done

