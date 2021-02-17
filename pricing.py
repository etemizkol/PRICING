# Bir oyun şirketi bir oyununda kullanıcılarına item satın
# alımları için hediye paralar vermiştir.
# Kullanıcılar bu sanal paraları kullanarak karakterlerine
# çeşitli araçlar satın almaktadır.
# Oyun şirketi bir item için fiyat belirtmemiş ve
# kullanıcılardan bu item'ı istedikleri fiyattan almalarını
# sağlamış.
# Örneğin kalkan isimli item için kullanıcılar kendi uygun
# gördükleri miktarları ödeyerek bu kalkanı satın alacaklar.
# Yani bir kullanıcı kendisine verilen sanal paralardan 30
# birim, diğer kullanıcı 45 birim ile ödeme yapabilir.
# Dolayısıyla kullanıcılar kendilerine göre ödemeyi göze
# aldıkları miktarlar ile bu item'ı satın alabilirler.


# Item'in fiyatı kategorilere göre farklılık göstermekte
# midir?

# Fiyat konusunda "hareket edebilir olmak"
# istenmektedir. Fiyat stratejisi için karar destek
# sistemi oluşturunuz.

# Olası fiyat değişiklikleri için item satın almalarını ve
# gelirlerini simüle ediniz.

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 20)
pd.set_option('expand_frame_repr', False) # KOLONLARIN / İLE AŞAĞI İNMESİNİ ENGELLER HEPSİ TEK SIRADA OLUR
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

df_=pd.read_csv(r"D:\DATA SCIENCE\VAHİT BAŞKAN\5. Hafta\PRICING_PROJECT\pricing.csv",sep=";")
df_.head()
df= df_.copy()
df.category_id.unique()
df.category_id.value_counts()
# Kategorilerin isimlerini anlaşılması daha kolay olması açısından temsili olarak değiştirelim
# Kategori isimlerini Knight Online oyunundaki serverlar olarak tanımladım
# ARES DIES BERAMUS CYPHER MANES EDENA
map = {489756:"ARES",874521:"DIES",361254:"BERAMUS",326584:"CYPHER",675201:"MANES",201436:"EDENA"}
server_names = ["ARES", "DIES", "BERAMUS", "CYPHER", "MANES", "EDENA"]
df.category_id.replace(map,inplace = True)
df.head()
df.info()
# nan bulunmamakta
df.groupby("category_id")["price"].agg(["mean","median","std","min","max"])
# Edena serveri hariç mean-medyan arasında büyük farklar bulunmakta

#------------------Deneme----------------------------
# listeye df atılabilir mi?
list=[]
    #list.append(df)
    #error verdi, atılamaz
# sözlüğe df atılabilir mi ?
dictionary={}
dictionary["Ares"]=df
# dictionarye atılabiliyor
print(dictionary)
#------------------Deneme bitti----------------------

# her bir kategoriyi ayrı ayrı df haline getirelim ve bir dictionarye atalım
all_categories = {}
for category in df.category_id.unique():
    all_categories[category] = df[df["category_id"] == category]

print(all_categories["ARES"].describe())
# ------------Deneme------------------------------
# Her server için ayrı ayrı describe atılması için

for key in all_categories:
    print(key)
    print(all_categories[key].describe([.01,.05,.25,.50,.95,.99]).T)
    print("-"*150)
# ------------Deneme Bitti------------------------------

# EDENA hariç diğer serverlarda outlier var.

servers = all_categories.copy()

# A/B testi için serverlerin ikili kombinasyonunun üretilmesi

combined_server_names = []
for first_server_index in range(len(server_names)):
    for second_server_index in range(first_server_index+1, len(server_names)):
        server_couple = [server_names[first_server_index], server_names[second_server_index]]
        combined_server_names.append(server_couple)

print(combined_server_names)

def is_norm_dist(sample):
    """ Verilen örneklemin normal dağılıp dağılmadığını % 95 güven aralığında kontrol eder
    Hipotez:
    H0 = Verilen örneklem dağılımı ile normal dağılım arasında istatistiki bir fark YOKTUR
    H1 = Verilen örneklem dağılımı ile normal dağılım arasında istatistiki bir fark VARDIR
    Girdiler:
    sample : ölçümlenecek örneklem (np series)
    Çıktılar:
    True - Normal Dağılıyor
    False - Normal Dağılmıyor"""

    from scipy.stats import shapiro
    test_statistic , pvalue = shapiro(sample)
    if pvalue < 0.05:
        # H0 Red, örneklem normal dağılmıyor
        return False
    else:
        # H0 Kabul, örneklem normal dağılıyor.
        return True

def is_vars_hom(sample1,sample2):
    """Verilen iki örneklemin varyanslarının aralarında homojen olup olmadığı levene test ile kontrol edilir.
    Hipotezler:
    H0: İki örneklemin dağılımları arasında istatistiki bir fark yoktur. u1  = u2
    H1: İki örneklemin dağılımları arasında istatistiki bir fark vardır. u1 != u2

    Girdiler:
    sample1 - birinci örneklem (np series)
    sample2 - ikinci örneklem (np series)

    Çıktılar:
    True - Örneklemlerin varyansı aynıdır.
    False - Örneklemlerin varyansı farklıdır.
    """
    from scipy import stats
    test_statistic , pvalue = stats.levene(sample1, sample2)
    if pvalue < 0.05:
        # H0 Red, varyanslar farklı
        return False
    else:
        # H0 Kabul, varyanslar aynı
        return True

def custom_ttest(sample1,sample2,equal_var = True):
    """
    Özelleştirilmiş bağımsız iki örneklem T testi.
    Hipotez:
    H0: İki grup ortalamaları arasında istatistiki olarak anlamlı bir farklılık yoktur.
    H1: İki grup ortalamaları arasında istatistiki olarak anlamlı bir farklılık vardır.
    Girdiler:
    :param sample1: 1. örneklem
    :param sample2: 2. örneklem
    :param equal_var: örneklem varyanslarının homojen dağılıp dağılmaması. True = Homojen, False = Homojen Değil
    :return:
    True- ortalamalar istatistiki olarak aynı
    False- ortalamalar istatistiki olarak farklı
    p_value - Hesaplanan p değeri.
    """
    from scipy import stats
    test_statistic,pvalue = stats.ttest_ind(sample1,sample2,equal_var=equal_var)
    if pvalue < 0.05:
        # H0 Red, ortalamalar istatistiki olarak farklı
        return False,pvalue
    else:
        # H0 Kabul, ortalamalar istatistiki olarak aynı
        return True,pvalue

def custom_manwhitneyu(sample1,sample2):
    """
    Özelleştirilmiş bağımsız mannwhitneyu testi.
    Hipotez:
    H0: İki grup ortalamaları arasında istatistiki olarak anlamlı bir farklılık yoktur.
    H1: İki grup ortalamaları arasında istatistiki olarak anlamlı bir farklılık vardır.
    Girdiler:
    :param sample1: 1. örneklem
    :param sample2: 2. örneklem
    :return:
    True- ortalamalar istatistiki olarak aynı
    False- ortalamalar istatistiki olarak farklı
    p_value - Hesaplanan p değeri.
    """
    from scipy import stats
    test_statistic, pvalue = stats.mannwhitneyu(sample1,sample2)
    if pvalue < 0.05:
        # H0 Red, ortalamalar istatistiki olarak farklı
        return False,pvalue
    else:
        # H0 Kabul, ortalamalar istatistiki olarak aynı
        return True,pvalue

def multiple_ab_analysis(combined_groups,target_col):
    final_df = pd.DataFrame()

    for combine_group in combined_groups:
        is_test_type_parametric = True
        p_value = 0
        test_result = True
        group_a = servers[combine_group[0]].loc[:,target_col]
        group_b = servers[combine_group[1]].loc[:,target_col]

        group_a_mean_median_count = [group_a.mean(),group_a.median(),group_a.count()]
        group_b_mean_median_count = [group_b.mean(), group_b.median(),group_b.count()]
        #print(group_b_mean_median_count,group_a_mean_median_count)

        # H0:İki örneklem ortalaması arasında fark yoktur. --> True döner
        # H1:İki örneklem ortalaması arasında fark vardır. --> False döner
        if is_norm_dist(group_a) and is_norm_dist(group_b):
            test_result,p_value = custom_ttest(group_a,group_b,equal_var=is_vars_hom(group_a,group_b))
            is_test_type_parametric = True
        else:
            test_result, p_value = custom_manwhitneyu(group_a,group_b)
            is_test_type_parametric = False

        # Result
        index_string = combine_group[0] + "-" + combine_group[1]
        temp = pd.DataFrame({"Grup Karşılaştırması" : "Fark Yok" if test_result else "Fark Var",
                             "pValue": p_value,
                             "A Grubu Normal Dağılıyor mu?":"Normal dağılıyor" if is_norm_dist(group_a) else "Normal Dağılmıyor",
                             "B Grubu Normal Dağılıyor mu?":"Normal dağılıyor" if is_norm_dist(group_b) else "Normal Dağılmıyor",
                             "Varyanslar Homojen Mi?":"Homojen" if is_vars_hom(group_a,group_b) else "Homojen Değil",
                             "Test Tipi":"Bağımsız 2 örneklem T Testi" if is_test_type_parametric else "Nonparametrik Man Whitney-U Testi",
                             "A Ortalama":group_a_mean_median_count[0],
                             "A Medyan":group_a_mean_median_count[1],
                             "A Gözlem Sayısı":group_a_mean_median_count[2],
                             "B Ortalama": group_b_mean_median_count[0],
                             "B Medyan": group_b_mean_median_count[1],
                             "B Gözlem Sayısı": group_b_mean_median_count[2],
                             },index=[index_string])


        col_order = ["Grup Karşılaştırması","Test Tipi","pValue","A Grubu Normal Dağılıyor mu?",
                     "B Grubu Normal Dağılıyor mu?","Varyanslar Homojen Mi?","A Ortalama","A Medyan","A Gözlem Sayısı",
                     "B Ortalama","B Medyan","B Gözlem Sayısı"]
        final_df = final_df.append(temp[col_order])

    return final_df


analyse_df = pd.DataFrame()
analyse_df = multiple_ab_analysis(combined_server_names,"price")
analyse_df.head(15)

# Görüldüğü gibi hiçbir grup normal dağılmıyor.
# Normal dağılıma yaklaşmak için outlier değerleri verisetimizden kaldıralım


def remove_outlier(df,column):
    """ (Q1 - 1.5 IQR) and (Q3 + 1.5 IQR) felsefesine dayanarak verilen kolondaki outlier değerler SİLİNECEK
    Argümanlar:
        df = işlem yapılacak dataframe objesi
        colums = aykırı değerleri kaldırılacak kolon
    Returns:
        Kırpılmış dataframe
    """

    Q1 = df[column].quantile(.25)
    Q3 = df[column].quantile(.75)
    IQR = (Q3-Q1)
    IQR_C = IQR * 1.5
    min_outlier = Q1 - IQR_C
    max_outlier = Q3 + IQR_C
    #df = df.drop(df[df.score < 50].index)
    return df[(df[column]>=min_outlier) & ((df[column]<=max_outlier))]

# Her bir sunucunun fiyat veirlerini kendi arasında hesaplayıp outlierları kaldırıyorum


for server_name,df in servers.items():
    servers[server_name] = remove_outlier(df,"price")


# Outlierlı Describe
for key in all_categories:
    print(key)
    print(all_categories[key].describe([.01,.05,.25,.50,.95,.99]).T)
    print("-"*150)

# Outliersız Describe
for key in servers:
    print(key)
    print(servers[key].describe([.01,.05,.25,.50,.95,.99]).T)
    print("-"*150)

analyse_df_non_outlier = multiple_ab_analysis(combined_server_names,"price")
analyse_df_non_outlier.head(15)

# Sonuçları karşılaştırdığımızda

# DIES, BERAMUS,MANES,EDENA serverlarında item fiyat ortalamaları aynı
# ARES, CYPHER serverlarında item fiyat ortalamaları hem yukarıdaki gruptan farklı, hemde kendi aralarında farklı.
# Yani toplamda 3 farklı fiyat ortalması grubu bulunmakta.
# 1. Grup: DIES, BERAMUS,MANES,EDENA
# 2. Grup: ARES
# 3. Grup: CYPHER

# -----Fiyatlandırma Politikaları---------
# Politika 1:
# Her sunucu için ayrı fiyatlandırma yapmak
# Politika 2:
# Her Grup için ayrı fiyatlandırma yapmak
# Politika 3:
# Grup 1 için ayrı, Grup 2 ve Grup 3'ü birleştirerek oluşacak yeni grup için ayrı fiyatlandırma yapmak

# Her politika için gelir tahmini yapılacaktır. Gelir tahmini yapılacakken kullanılacak parametreler şunlardır:
# Mod
# Medyan
# Ortalama
# Güven Aralığı alt sınır
# Güven Aralığı orta
# Güven Aralığı üst sınır


# Fiyatlandırma opsiyonları oluşturabilmek için ana veri setini yukarıdaki üç gruba göre bölüyorum
# İşlemlerimi outlierları ayıklanmış verilerden devam ettiriyorum çünkü piyasayı manipüle edecek hareketleri yok sayıyorum.
df_.category_id.unique()
# 1. Grup: DIES, BERAMUS,MANES,EDENA
group_1_list = ["DIES","BERAMUS","MANES","EDENA"]
group_1 = pd.DataFrame()
for group_1_item in group_1_list:
    group_1 = group_1.append(servers[group_1_item])
# 2. Grup: ARES
group_2 =servers["ARES"]
# 3. Grup: CYPHER
group_3 =servers["CYPHER"]


# Gelir simulasyonu, (sunucuda satınalma yapan kişi sayısı * tahmini fiyat) operasyonu ile hesaplanacaktır.
# Mikroekonomideki arz ve talep kanunlarına göre,
# bir ürünün fiyatı azaldıkça ona karşı oluşan talep artar,
# fiyat arttıkça ona karşı oluşan talep azalır.
# Buna dayanarak güven aralığı üst limit fiyatlandırmasında %5 müşteri eklenecek,
# alt limit fiyatlandırasında %5 müşteri çıkartılacaktır.
# Kaynak : https://tr.wikipedia.org/wiki/Arz_ve_talep

# ----------------------------------------------
# Politika 1: Her sunucu için ayrı fiyatlandırma
# ----------------------------------------------

policy_1_mean = 0
for name_ in server_names:
    policy_1_mean = float(servers[name_].mean()) * servers[name_]["price"].count() + policy_1_mean

policy_1_median = 0
for name_ in server_names:
    policy_1_median = float(servers[name_].median()) * servers[name_]["price"].count() + policy_1_median

policy_1_ci_low = 0
import statsmodels.stats.api as sms
for name_ in server_names:
    # (alt_limit,üst_limit)
    # Güven aralığı yorumu:
    # İtem satınalma fiyatı %95 güven aralığında hesaplanan değerler arasında olacaktır.
    # 100 müşterinin 95'i bu item'a bu aralıkta fiyat verecektir.
    # Düşük fiyat olduğu için  +%5 müşteri eklenecek
    # print(sms.DescrStatsW(servers[name_]["price"]).tconfint_mean()[0])
    policy_1_ci_low = policy_1_ci_low + sms.DescrStatsW(servers[name_]["price"]).tconfint_mean()[0] * servers[name_]["price"].count() * 1.05

policy_1_ci_medium = 0
for name_ in server_names:
    # (alt_limit,üst_limit)
    # Güven aralığı yorumu:
    # İtem satınalma fiyatı %95 güven aralığında hesaplanan değerler arasında olacaktır.
    # 100 müşterinin 95'i bu item'a bu aralıkta fiyat verecektir.
    # print(sms.DescrStatsW(servers[name_]["price"]).tconfint_mean()[0])
    ci_mean = (sms.DescrStatsW(servers[name_]["price"]).tconfint_mean()[1]+sms.DescrStatsW(servers[name_]["price"]).tconfint_mean()[0])/2
    policy_1_ci_medium = policy_1_ci_medium + ci_mean * servers[name_]["price"].count()


policy_1_ci_high = 0
for name_ in server_names:
    # (alt_limit,üst_limit)
    # Güven aralığı yorumu:
    # İtem satınalma fiyatı %95 güven aralığında hesaplanan değerler arasında olacaktır.
    # 100 müşterinin 95'i bu item'a bu aralıkta fiyat verecektir.
    # Düşük fiyat olduğu için  -%10 müşteri eklenecek
    # print(sms.DescrStatsW(servers[name_]["price"]).tconfint_mean()[0])
    policy_1_ci_high = policy_1_ci_high + sms.DescrStatsW(servers[name_]["price"]).tconfint_mean()[1] * servers[name_]["price"].count() * 0.95

# ----------------------------------------------
# Politika 2 Her Grup için ayrı fiyatlandırma
# ----------------------------------------------

# Ortalama
policy_2_mean_group_1 = group_1.price.mean() * group_1.price.count()
policy_2_mean_group_2 = group_2.price.mean() * group_2.price.count()
policy_2_mean_group_3 = group_3.price.mean() * group_3.price.count()
policy_2_mean_total = policy_2_mean_group_1 + policy_2_mean_group_2 + policy_2_mean_group_3
# Medyan
policy_2_median_group_1 = group_1.price.median() * group_1.price.count()
policy_2_median_group_2 = group_2.price.median() * group_2.price.count()
policy_2_median_group_3 = group_3.price.median() * group_3.price.count()
policy_2_median_total = policy_2_median_group_1 + policy_2_median_group_2 + policy_2_median_group_3
# Güven aralığı Alt
policy_2_ci_low_group_1 = sms.DescrStatsW(group_1["price"]).tconfint_mean()[0] * group_1.price.count()*1.05
policy_2_ci_low_group_2 = sms.DescrStatsW(group_2["price"]).tconfint_mean()[0] * group_2.price.count()*1.05
policy_2_ci_low_group_3 = sms.DescrStatsW(group_3["price"]).tconfint_mean()[0] * group_3.price.count()*1.05
policy_2_ci_low_total = policy_2_ci_low_group_1 + policy_2_ci_low_group_2 + policy_2_ci_low_group_3
# Güven aralığı Orta
policy_2_ci_med_group_1 = (sms.DescrStatsW(group_1["price"]).tconfint_mean()[0]+sms.DescrStatsW(group_1["price"]).tconfint_mean()[1])/2 * group_1.price.count()
policy_2_ci_med_group_2 = (sms.DescrStatsW(group_2["price"]).tconfint_mean()[0]+sms.DescrStatsW(group_2["price"]).tconfint_mean()[1])/2 * group_2.price.count()
policy_2_ci_med_group_3 = (sms.DescrStatsW(group_3["price"]).tconfint_mean()[0]+sms.DescrStatsW(group_3["price"]).tconfint_mean()[1])/2 * group_3.price.count()
policy_2_ci_med_total = policy_2_ci_med_group_1 + policy_2_ci_med_group_2 + policy_2_ci_med_group_3
# Güven aralığı Üst
policy_2_ci_high_group_1 = sms.DescrStatsW(group_1["price"]).tconfint_mean()[1] * group_1.price.count()*0.95
policy_2_ci_high_group_2 = sms.DescrStatsW(group_2["price"]).tconfint_mean()[1] * group_2.price.count()*0.95
policy_2_ci_high_group_3 = sms.DescrStatsW(group_3["price"]).tconfint_mean()[1] * group_3.price.count()*0.95
policy_2_ci_high_total = policy_2_ci_high_group_1 + policy_2_ci_high_group_2 + policy_2_ci_high_group_3

# --------------------------------------------------------
# Politika 3 Grup 1 ayrı, Grup 2 ve 3 birleştirilerek ayrı
# --------------------------------------------------------
group_2_3 = group_2.append(group_3)
# Ortalama
policy_3_mean_group_1 = group_1.price.mean() * group_1.price.count()
policy_3_mean_group_2_3 = group_2_3.price.mean() * group_2_3.price.count()
policy_3_mean_total = policy_3_mean_group_1 + policy_3_mean_group_2_3
# Medyan
policy_3_median_group_1 = group_1.price.median() * group_1.price.count()
policy_3_median_group_2_3 = group_2_3.price.median() * group_2_3.price.count()
policy_3_median_total = policy_3_median_group_1 + policy_3_median_group_2_3
# Güven aralığı alt
policy_3_ci_low_group_1 = sms.DescrStatsW(group_1["price"]).tconfint_mean()[0] * group_1.price.count()*1.05
policy_3_ci_low_group_2_3 = sms.DescrStatsW(group_2_3["price"]).tconfint_mean()[0] * group_2_3.price.count()*1.05
policy_3_ci_low_total = policy_3_ci_low_group_1 + policy_3_ci_low_group_2_3
# Güven Aralığı ortalama
policy_3_ci_med_group_1 = (sms.DescrStatsW(group_1["price"]).tconfint_mean()[0]+sms.DescrStatsW(group_1["price"]).tconfint_mean()[1])/2 * group_1.price.count()
policy_3_ci_med_group_2_3 = (sms.DescrStatsW(group_2_3["price"]).tconfint_mean()[0]+sms.DescrStatsW(group_2_3["price"]).tconfint_mean()[1])/2 * group_2_3.price.count()
policy_3_ci_med_total = policy_3_ci_med_group_1 + policy_3_ci_med_group_2_3
# Güven aralığı üst
policy_3_ci_high_group_1 = sms.DescrStatsW(group_1["price"]).tconfint_mean()[1] * group_1.price.count()*0.95
policy_3_ci_high_group_2_3 = sms.DescrStatsW(group_2_3["price"]).tconfint_mean()[1] * group_2_3.price.count()*0.95
policy_3_ci_high_total = policy_3_ci_high_group_1 + policy_3_ci_high_group_2_3



result_df_dict = {"Politika":["Politika 1","Politika 2","Politika 3"],
             "Mean":[policy_1_mean,policy_2_mean_total,policy_3_mean_total],
             "Median":[policy_1_median,policy_2_median_total,policy_3_median_total],
             "Confidence Interval, Low":[policy_1_ci_low,policy_2_ci_low_total,policy_3_ci_low_total],
             "Confidence Interval, Medium":[policy_1_ci_medium,policy_2_ci_med_total,policy_3_ci_med_total],
             "Confidence Interval, High":[policy_1_ci_high,policy_2_ci_high_total,policy_3_ci_high_total]}
# karar destek matrisi:
result = pd.DataFrame(result_df_dict)











