import sys
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

# --- КОНФИГУРАЦИЯ ---
EVENTS_PATH = "/user/master/data/geo/events"
GEO_PATH = "/user/s26586465/data/geo/geo.csv"
OUTPUT_PATH = "/user/s26586465/analytics/user_mart"

def get_spark_session():
    return SparkSession.builder \
        .appName("UserMart_Final_Ultra_Safe") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

def load_geo(spark):
    """Загрузка городов с безопасным сопоставлением таймзон штатов"""
    # В Австралии зоны делятся по штатам. Мы привяжем города к основным стабильным зонам.
    return spark.read.csv(GEO_PATH, sep=';', header=True) \
        .withColumn("city_lat", F.radians(F.regexp_replace(F.col("lat"), ",", ".").cast("double"))) \
        .withColumn("city_lon", F.radians(F.regexp_replace(F.col("lng"), ",", ".").cast("double"))) \
        .withColumn("timezone", 
            F.when(F.col("city").isin("Sydney", "Newcastle", "Wollongong", "Central Coast"), "Australia/Sydney")
             .when(F.col("city").isin("Melbourne", "Geelong", "Ballarat"), "Australia/Melbourne")
             .when(F.col("city").isin("Brisbane", "Gold Coast", "Rockhampton", "Townsville", "Cairns"), "Australia/Brisbane")
             .when(F.col("city").isin("Adelaide"), "Australia/Adelaide")
             .when(F.col("city").isin("Perth"), "Australia/Perth")
             .when(F.col("city").isin("Darwin"), "Australia/Darwin")
             .when(F.col("city").isin("Hobart"), "Australia/Hobart")
             .when(F.col("city").isin("Canberra"), "Australia/Canberra")
             # Если города нет в списке, используем Сидней как дефолт, чтобы скрипт не падал
             .otherwise("Australia/Sydney")
        ) \
        .select("city", "city_lat", "city_lon", "timezone")

def load_events(spark, target_date='2022-05-01'):
    """Загрузка событий строго за ОДИН день"""
    return spark.read.parquet(EVENTS_PATH) \
        .filter(F.col("date") == target_date) \
        .withColumn("user_id", F.col("event.message_from")) \
        .filter(F.col("lat").isNotNull() & F.col("lon").isNotNull() & F.col("user_id").isNotNull()) \
        .withColumn("msg_lat", F.radians(F.col("lat"))) \
        .withColumn("msg_lon", F.radians(F.col("lon"))) \
        .select("user_id", "date", "msg_lat", "msg_lon", F.col("event.datetime").alias("TIME_UTC"))

def main():
    spark = get_spark_session()
    
    # 1. Загрузка данных (Берем 1 мая 2022 года)
    geo = load_geo(spark)
    events = load_events(spark, target_date='2022-05-01')

    # 2. Расчет ближайшего города
    dist_expr = 2 * 6371 * F.asin(
        F.sqrt(
            F.pow(F.sin((F.col("msg_lat") - F.col("city_lat")) / 2), 2) +
            F.cos(F.col("msg_lat")) * F.cos(F.col("city_lat")) * F.pow(F.sin((F.col("msg_lon") - F.col("city_lon")) / 2), 2)
        )
    )
    
    w_dist = Window.partitionBy("user_id", "TIME_UTC").orderBy("distance")
    
    actual_events = events.crossJoin(F.broadcast(geo)) \
        .withColumn("distance", dist_expr) \
        .withColumn("rn", F.row_number().over(w_dist)) \
        .filter("rn = 1") \
        .select("user_id", "date", "city", "timezone", "TIME_UTC")

    # 3. act_city и местное время
    w_user = Window.partitionBy("user_id").orderBy(F.desc("TIME_UTC"))
    
    user_info = actual_events.withColumn("rn", F.row_number().over(w_user)) \
        .filter("rn = 1") \
        .withColumn("local_time", F.from_utc_timestamp(F.col("TIME_UTC"), F.col("timezone"))) \
        .select(F.col("user_id"), F.col("city").alias("act_city"), "local_time")

    # 4. travel_stats
    w_travel = Window.partitionBy("user_id").orderBy("TIME_UTC")
    
    travel_stats = actual_events.withColumn("prev_city", F.lag("city").over(w_travel)) \
        .filter(F.col("prev_city").isNull() | (F.col("city") != F.col("prev_city"))) \
        .groupBy("user_id").agg(
            F.count("city").alias("travel_count"), 
            F.collect_list("city").alias("travel_array")
        )

    # 5. home_city (за 1 день он может не рассчитаться, но логику оставим)
    home_city = actual_events.withColumn("is_change", F.when(F.col("city") != F.lag("city").over(w_travel), 1).otherwise(0)) \
        .withColumn("group", F.sum("is_change").over(w_travel)) \
        .groupBy("user_id", "city", "group").agg(
            F.min("date").alias("start"), 
            F.max("date").alias("end")
        ) \
        .withColumn("days", F.datediff("end", "start")) \
        .filter("days >= 27") \
        .withColumn("rn_h", F.row_number().over(Window.partitionBy("user_id").orderBy(F.desc("end")))) \
        .filter("rn_h = 1").select(F.col("user_id"), F.col("city").alias("home_city"))

    # 6. Финальный Join и сохранение
    final_mart = user_info \
        .join(home_city, "user_id", "left") \
        .join(travel_stats, "user_id", "left")
    
    final_mart.coalesce(1).write.mode("overwrite").parquet(OUTPUT_PATH)
    
    print(f"--- SUCCESS: Витрина за 1 день сохранена в {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()