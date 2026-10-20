import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

"""
基础数据观察
几个点：
1、order_amount最小值是0 --> 0元订单（80条,占比0.11%），
   推测为赠品/促销单；因金额为0不影响金额类指标加总，且占比极低，故保留不处理
2、平均每人下单数不到3单,是不是有很多人不会回购?
3、订单金额的平均值约为36,但是中位数仅为26,说明有许多大额订单拉高了平均值?
"""
df = pd.read_csv('CDNOW.txt', sep=r'\s+',
                 names=['user_id', 'order_dt', 'order_num', 'order_amount'])
df['order_dt'] = pd.to_datetime(df['order_dt'], format='%Y%m%d')

# 数据核查：0元订单的数量、占比、下单件数分布（用于判断是否异常，不影响后续计算）
zero_orders = df[df['order_amount'] == 0]
print(f"0元订单：{len(zero_orders)} 条，占比 {len(zero_orders) / len(df):.2%}")
print(f"0元订单的下单件数分布：\n{zero_orders['order_num'].describe()}")

print('基础信息:')
print(df.head(5))
print("数据形状：", df.shape)
print("时间范围：", df['order_dt'].min(), "至", df['order_dt'].max())
print("基础统计信息：", df.describe())
print("用户数：", df['user_id'].nunique())
print("总消费金额：{:.2f}".format(df['order_amount'].sum()))
print("缺失值：\n", df.isnull().sum())
print()

"""
part1  整体运营情况？
每月订单量,销售额 变化？
运行完以后会发现：订单量和销售额都是暴涨以后回落 -- 为什么?
"""
df['month'] = df['order_dt'].dt.to_period('M')
df_monthly = df.groupby('month')[['order_num', 'order_amount']].sum()
df_monthly[['order_num', 'order_amount']].plot(subplots=True)
plt.show()

"""
part2  用户个体什么水平？
每个用户总共花多少钱、买几次？分布什么样？
"""

user = df.groupby('user_id').agg(
    total_amount=('order_amount', 'sum'),
    order_count=('user_id', 'count')
)
# user['total_amount'].hist(bins=50)  大多集中在左半边
print('用户消费总金额统计信息')
print(user['total_amount'].describe())

user_amount = df.groupby('user_id')['order_amount'].sum().sort_values(ascending=False)
total_amount = user_amount.sum()
total_users = len(user_amount)


def top_users_share(user_amount, top_pct):
    n = max(1, int(len(user_amount) * top_pct))  # 前N%的用户数
    share = user_amount.head(n).sum() / user_amount.sum()
    return n, share


print()
for pct in [0.01, 0.05, 0.10, 0.20, 0.50]:
    n, share = top_users_share(user_amount, pct)

    print(f"前{pct:.0%}的用户（{n}人）贡献了{share:.1%}的消费金额")

print()
plt.rcParams['font.sans-serif'] = ['SimHei']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
user[user['total_amount'] < 500]['total_amount'].hist(bins=50, ax=axes[0])
axes[0].set_title('消费金额分布')
axes[0].set_xlabel('消费总金额')
axes[0].set_ylabel('人数')

user[user['order_count'] < 40]['order_count'].hist(bins=50, ax=axes[1])
axes[1].set_title('消费次数分布')
axes[1].set_xlabel('消费次数')
axes[1].set_ylabel('次数')

plt.tight_layout()
plt.show()

"""
part3 RFM模型
"""
now = df['order_dt'].max() + pd.Timedelta(days=1)
user['R'] = (now - df.groupby('user_id')['order_dt'].max()).dt.days
user['R_score'] = (user['R'] < user['R'].mean()).astype(int)
user['F_score'] = (user['order_count'] > user['order_count'].mean()).astype(int)
user['M_score'] = (user['total_amount'] > user['total_amount'].mean()).astype(int)
user['RFM'] = user['R_score'].astype(str) + user['F_score'].astype(str) + user['M_score'].astype(str)
rfm_map = {
    '111': '重要价值用户',
    '110': '重要深耕用户',
    '101': '重要唤回用户',
    '100': '重要发展用户',
    '011': '重要保持用户',
    '010': '重要挽留用户',
    '001': '一般挽留用户',
    '000': '一般发展用户'
}
user['user_type'] = user['RFM'].map(rfm_map)
print(user['user_type'].value_counts())

"""
part4 复购率 & 回购率
"""
repurchase_rate = (user['order_count'] >= 2).sum() / len(user)
print()
print(f'复购率：{repurchase_rate:.2%}')
pivot = df.pivot_table(index='user_id', columns='month', values='order_amount', aggfunc='count').notna().astype(int)
repurchase_monthly = (pivot & pivot.shift(axis=1)).sum() / pivot.shift(axis=1).sum()
repurchase_monthly.plot()
plt.show()


"""
part5 复购用户 vs 一次性用户，客单价有没有显著差异?
"""

user_stats = df.groupby('user_id').agg(
    total_amount=('order_amount', 'sum'),
    order_count=('user_id', 'count')
)
user_stats['aov'] = user_stats['total_amount'] / user_stats['order_count']

repurchase_group = user_stats[user_stats['order_count'] >= 2]['aov']
onetime_group = user_stats[user_stats['order_count'] == 1]['aov']

# t检验包括：单样本t检验，独立样本t检验(Student/Welch)，配对样本t检验，比较两组不同的人用独立样本t检验，默认用Student's t检验
# Student's t检验 的前提一是要求数据近似正太分布，二是两组数据的方差（离散程度）相差不多
t_stat, p_value = stats.ttest_ind(repurchase_group, onetime_group,equal_var=False)
print()
print(f'复购组均值:{repurchase_group.mean():.4f}')
print(f'一次性组均值：{onetime_group.mean():.4f}')
print(f't统计值:{t_stat:.4f}')
print(f'p值:{p_value:.4f}')

if p_value < 0.05:
    print('差异显著:复购用户和一次性用户的客单价存在真实差异')
else:
    print('差异不明显：现有数据无法证明两组客单价有差异')

"""
part6 高消费和低消费用户，复购率有没有显著差异?
"""
user_stats['spend_level'] = pd.cut(user_stats['total_amount'],
                                   bins=[-1, 100, float('inf')],
                                   labels=['低消费', '高消费'])
user_stats['repurchase_status'] = (user_stats['order_count'] >= 2).map({True: '复购', False: '未复购'})
ct = pd.crosstab(user_stats['spend_level'], user_stats['repurchase_status'])
chi2, p_value, dof, expected = chi2_contingency(ct)
print()
print(f'卡方统计值:{chi2:.4f}')
print(f'p值:{p_value:.4f}')
if p_value < 0.05:
    print('差异显著:高消费和低消费用户，复购率存在显著差异')
else:
    print('差异不明显:现有数据无法证明不同消费用户的复购率之间有显著差异')
