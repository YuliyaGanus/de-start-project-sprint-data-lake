import sys
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

# --- КОНФИГУРАЦИЯ ---
EVENTS_PATH = "/user/master/data/geo/events"
GEO_PATH = "/user/s26586465/data/geo/geo.csv"
OUTPUT_PATH = "/user/s26586465/analytics/geo_stats"

def main():
    spark = SparkSession.builder \
        .appName("GeoZones_Fixed_Schema") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    # 1. Загрузка справочника городов
    geo = spark.read.csv(GEO_PATH, sep=';', header=True) \
        .withColumn("city_lat", F.radians(F.regexp_replace(F.col("lat"), ",", ".").cast("double"))) \
        .withColumn("city_lon", F.radians(F.regexp_replace(F.col("lng"), ",", ".").cast("double"))) \
        .select(F.col("id").alias("zone_id"), "city_lat", "city_lon")

    # 2. Загрузка событий (1 день)
    events = spark.read.parquet(EVENTS_PATH).filter("date == '2022-05-01'")

    # Исправленная разметка на основе твоего лога:
    # Используем subscription_user вместо subscription_from
    geo_events = events.select(
        F.coalesce(
            F.col("event.message_from"), 
            F.col("event.reaction_from"), 
            F.col("event.subscription_user") # Исправлено здесь
        ).alias("user_id"),
        F.when(F.col("event.message_id").isNotNull(), 1).otherwise(0).alias("is_msg"),
        F.when(F.col("event.reaction_from").isNotNull(), 1).otherwise(0).alias("is_react"),
        F.when(F.col("event.subscription_user").isNotNull(), 1).otherwise(0).alias("is_sub"),
        F.radians(F.col("lat")).alias("lat"),
        F.radians(F.col("lon")).alias("lon"),
        F.col("date")
    ).filter("lat IS NOT NULL AND lon IS NOT NULL")

    # 3. Привязка к городам (Гаверсинус)
    dist_expr = 2 * 6371 * F.asin(F.sqrt(
        F.pow(F.sin((F.col("lat") - F.col("city_lat")) / 2), 2) +
        F.cos(F.col("lat")) * F.cos(F.col("city_lat")) * F.pow(F.sin((F.col("lon") - F.col("city_lon")) / 2), 2)
    ))

    w_dist = Window.partitionBy("user_id", "date", "lat", "lon").orderBy("distance")
    
    events_with_zone = geo_events.crossJoin(F.broadcast(geo)) \
        .withColumn("distance", dist_expr) \
        .withColumn("rn", F.row_number().over(w_dist)) \
        .filter("rn = 1") \
        .select("user_id", "zone_id", "date", "is_msg", "is_react", "is_sub") \
        .withColumn("month", F.trunc(F.col("date"), "month")) \
        .withColumn("week", F.trunc(F.col("date"), "week"))

    # 4. Регистрации
    reg_window = Window.partitionBy("user_id").orderBy("date")
    registrations = events_with_zone.withColumn("rn_reg", F.row_number().over(reg_window)) \
        .filter("rn_reg = 1") \
        .groupBy("month", "week", "zone_id").agg(F.count("user_id").alias("week_user"))

    # 5. Агрегация недель
    weekly = events_with_zone.groupBy("month", "week", "zone_id").agg(
        F.sum("is_msg").alias("week_message"),
        F.sum("is_react").alias("week_reaction"),
        F.sum("is_sub").alias("week_subscription")
    )

    # 6. Финальная витрина через окна
    w_month = Window.partitionBy("month", "zone_id")
    res = weekly.join(registrations, ["month", "week", "zone_id"], "left").fillna(0) \
        .withColumn("month_message", F.sum("week_message").over(w_month)) \
        .withColumn("month_reaction", F.sum("week_reaction").over(w_month)) \
        .withColumn("month_subscription", F.sum("week_subscription").over(w_month)) \
        .withColumn("month_user", F.sum("week_user").over(w_month))

    # Сохранение
    res.coalesce(1).write.mode("overwrite").parquet(OUTPUT_PATH)
    print("--- SUCCESS: ГЕО-ВИТРИНА ПОСТРОЕНА ---")

if __name__ == "__main__":
    main()