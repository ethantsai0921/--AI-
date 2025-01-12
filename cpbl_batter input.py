import pandas as pd
import cpbl_batter

cpbl_csv = pd.read_csv(filepath_or_buffer=r"C:\Users\user\Downloads\CPBL_batting_stats_precision_use.csv",encoding="utf_8",sep=",")
user = input("請輸入要預測的值: (#h,#avg,#bb,#g,#hr,#obp,#pa,#rbi,#r,#slg)" "\n" )
playername = input("輸入要查詢的選手: ")

if user == 'h' and  (playername in cpbl_csv['player'].values):
  print(cpbl_batter.h(playername))
elif user == 'avg' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.avg(playername))
elif user == 'bb' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.bb(playername))
elif user == 'g' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.g(playername))
elif user == 'hr' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.hr(playername))
elif user == 'obp' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.obp(playername))
elif user == 'pa' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.pa(playername))
elif user == 'rbi' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.rbi(playername))
elif user == 'r' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.r(playername))
elif user == 'slg' and (playername in cpbl_csv['player'].values):
  print(cpbl_batter.slg(playername))
else:
  print('error')


