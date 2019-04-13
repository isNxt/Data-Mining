import pandas as pd
import matplotlib.pyplot as plt

'''
1. Analysis

實驗使用2.2 敘述之資料集，比較兩種演算法在改變 minsup 數值以及資料量的情況下其時間的變化。 
其中 FP-Growth 之結果為十次運算之時間平均，Aprior 由於時間上之考量為兩次運算之時間平均。

'''
kosarak = pd.read_csv('./kosarak-report.csv')
transactions = pd.read_csv('./transactions-report.csv')
'''
a. kosarak

在 IBM 合成資料集中，實驗使用四種不同 item 數，與資料量數之資料，結果顯示在 Figure 1 ~ Figure 4 中。 
minsup由於不同資料集所含 frequent patterns 數目不同，因此有 0.5% ~ 0.9%, 0.2% ~ 0.5% 兩種設定。 
在實驗中可以觀察到 FP growth 的效率遠遠超過 Apriori。 而當 minsup 值升高時兩者的時間差則開始縮小，
其原因是因為 minsup 值升高則frequent pattern 的數目減少，而 Apriori 所需要 join 與搜索的 Candicates 數目也隨之減少的緣故。 

改變 Item 種類數目 
而從 Figure 3 跟 Figure 5 可以看到item 種類增加(5000~30000)，Apriori 時間似乎會花較久，但事實上是因為後者生成之的FP數目較多，Candidate 數目也較多所導致。 

改變 Datasize 
Figure 5 與 Figure 7 則是改變 Transaction 數目，從約10000筆資料到 20000筆資料。 雖然 Apriori 需要不斷 Scan 整份資料集，
但可以看到在目前的資料量下，Apriori 依然是FP 數目影響較大，FP 越多，花的時間越多， 
但 FP growth 就不同了，雖然圖片看來差異不大，但在20000筆資料所花的時間是略為大於10000筆資料的。
'''
plt.figure(figsize=(20, 10))

plt.subplot(121)
x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
y_1 = kosarak['time'][0:14]
y_2 = kosarak['time'][14:28]
plt.plot(x, y_1, 'bo-', alpha=0.5, lw=3)
plt.plot(x, y_2, 'ro-', alpha=0.5)
plt.xlabel('minsup (%)')
plt.ylabel('time (sec)')
plt.legend(["FP Growth", "Apriori"], loc=1)
plt.title('Figure 1: kosarak running time')

plt.subplot(122)
x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
y_1 = kosarak['fp_num'][0:14]
plt.plot(x, y_1, 'bo-', alpha=0.5, lw=3)
plt.xlabel('minsup (%)')
plt.ylabel('number of frequent pattern')
plt.title('Figure 2: kosarak fp number')

plt.savefig('kosarak.png')
plt.show()

rules = pd.read_csv('../rules/kosarak-rules_fp_0.01.csv')
print(rules.head(10))

'''
b.transactions
'''
print(transactions.head())
'''
觀察 Figure 10 與 Figure 11，在 Kaggle 資料集 Apriori 依舊比 FP-growth 慢許多，
雖然資料僅有 1499 筆，但平均每個 Transaction 有 15 項 Item，算法需要花費極大的Join 與 prune 時間，
為此即便FP 數目少 Apriori 所花費時間仍為IBM 1000 筆資料時的 20倍。
'''
plt.figure(figsize=(20, 10))

plt.subplot(121)
x = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
y_1 = transactions['time'][0:19]
y_2 = transactions['time'][19:38]
plt.plot(x, y_1, 'bo-', alpha=0.5, lw=3)
plt.plot(x, y_2, 'ro-', alpha=0.5)
plt.xlabel('minsup (%)')
plt.ylabel('time (sec)')
plt.legend(["FP Growth", "Apriori"], loc=1)
plt.title('Figure 3: transactions running time')

plt.subplot(122)
x = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
y_1 = transactions['fp_num'][0:19]
plt.plot(x, y_1, 'bo-', alpha=0.5, lw=3)
plt.xlabel('minsup (%)')
plt.ylabel('number of frequent pattern')
plt.title('Figure 4: transactions fp number')

plt.show()
plt.savefig('transactions.png')
plt.show()

rules = pd.read_csv('../rules/transactions-rules_fp_0.01.csv')
print(rules.head(10))

'''
2. Conclusion

FP-Growth 算法比起 Apriori 有顯著的效率差異，
原因是它透過建立 FP-Tree 來減少探索時間，相對的 Apriori 則需要不斷 Scan 資料集來確認是否滿足 minisup 條件。 
雖然這種FP-Tree的結構在資料量大時仍具有優勢，
但當 frequent patterns 的數目減少時(minsup增加)，兩者的時間差異會越來越少，這時建立 FP-Tree 這種資料結構的時間花費便體現出來。
'''