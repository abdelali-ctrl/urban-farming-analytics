
## 1. Basic Queries

### 1.1 Total Yield by Crop Type
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    [Dim Crop].[Crop Type].Members ON ROWS
FROM [Urban Farming DW]
```

### 1.2 Production Count by Region
```mdx
SELECT 
    [Measures].[Fact Production Count] ON COLUMNS,
    [Dim Region].[Region Name].Members ON ROWS
FROM [Urban Farming DW]
```

### 1.3 All Measures Overview
```mdx
SELECT 
    {
        [Measures].[Yield Kg Per Hectare],
        [Measures].[Yield Per Day],
        [Measures].[Water Efficiency],
        [Measures].[Soil Moisture Pct],
        [Measures].[Soil p H],
        [Measures].[Growing Conditions Score],
        [Measures].[Fact Production Count]
    } ON COLUMNS
FROM [Urban Farming DW]
```

---

## 2. Cross-Dimensional Analysis

### 2.1 Yield by Crop and Region
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    NON EMPTY CROSSJOIN(
        [Dim Crop].[Crop Type].Members,
        [Dim Region].[Region Name].Members
    ) ON ROWS
FROM [Urban Farming DW]
```

### 2.2 Water Efficiency by Irrigation Type and Crop
```mdx
SELECT 
    [Measures].[Water Efficiency] ON COLUMNS,
    NON EMPTY CROSSJOIN(
        [Dim Irrigation].[Irrigation Type].Members,
        [Dim Crop].[Crop Type].Members
    ) ON ROWS
FROM [Urban Farming DW]
```

### 2.3 Yield by Season and Crop Type
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    NON EMPTY CROSSJOIN(
        [Harvest Time].[Season].Members,
        [Dim Crop].[Crop Type].Members
    ) ON ROWS
FROM [Urban Farming DW]
```

---

## 3. Time-Based Analysis

### 3.1 Yield by Harvest Quarter
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    [Harvest Time].[Quarter].Members ON ROWS
FROM [Urban Farming DW]
```

### 3.2 Production by Sowing Month
```mdx
SELECT 
    {[Measures].[Yield Kg Per Hectare], [Measures].[Fact Production Count]} ON COLUMNS,
    [Sowing Time].[Month Name].Members ON ROWS
FROM [Urban Farming DW]
```

### 3.3 Seasonal Yield Comparison
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    [Harvest Time].[Season].Members ON ROWS
FROM [Urban Farming DW]
```

---

## 4. Filtering with WHERE Clause

### 4.1 Yield for Specific Crop Type
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    [Dim Region].[Region Name].Members ON ROWS
FROM [Urban Farming DW]
WHERE [Dim Crop].[Crop Type].&[Tomato]
```

### 4.2 Water Efficiency for Drip Irrigation
```mdx
SELECT 
    {[Measures].[Water Efficiency], [Measures].[Yield Kg Per Hectare]} ON COLUMNS,
    [Dim Crop].[Crop Type].Members ON ROWS
FROM [Urban Farming DW]
WHERE [Dim Irrigation].[Irrigation Type].&[Drip]
```

### 4.3 Production with No Disease
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    [Dim Crop].[Crop Type].Members ON ROWS
FROM [Urban Farming DW]
WHERE [Dim Disease].[Disease Status].&[None]
```

---

## 5. Top N Analysis

### 5.1 Top 5 Regions by Yield
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    TOPCOUNT(
        [Dim Region].[Region Name].Members,
        5,
        [Measures].[Yield Kg Per Hectare]
    ) ON ROWS
FROM [Urban Farming DW]
```

### 5.2 Top 3 Crops by Water Efficiency
```mdx
SELECT 
    [Measures].[Water Efficiency] ON COLUMNS,
    TOPCOUNT(
        [Dim Crop].[Crop Type].Members,
        3,
        [Measures].[Water Efficiency]
    ) ON ROWS
FROM [Urban Farming DW]
```

### 5.3 Bottom 5 Regions by Growing Conditions Score
```mdx
SELECT 
    [Measures].[Growing Conditions Score] ON COLUMNS,
    BOTTOMCOUNT(
        [Dim Region].[Region Name].Members,
        5,
        [Measures].[Growing Conditions Score]
    ) ON ROWS
FROM [Urban Farming DW]
```

---

## 6. Calculated Members

### 6.1 Total Production Metrics with Average pH
```mdx
WITH 
    MEMBER [Measures].[Avg Soil pH] AS 
        AVG([Dim Region].[Region Name].Members, [Measures].[Soil p H])
SELECT 
    {[Measures].[Yield Kg Per Hectare], [Measures].[Avg Soil pH]} ON COLUMNS,
    [Dim Crop].[Crop Type].Members ON ROWS
FROM [Urban Farming DW]
```

### 6.2 Yield Efficiency Ratio (Yield per Water Usage)
```mdx
WITH 
    MEMBER [Measures].[Yield to Moisture Ratio] AS 
        IIF(
            [Measures].[Soil Moisture Pct] = 0,
            NULL,
            [Measures].[Yield Kg Per Hectare] / [Measures].[Soil Moisture Pct]
        )
SELECT 
    {[Measures].[Yield Kg Per Hectare], [Measures].[Yield to Moisture Ratio]} ON COLUMNS,
    [Dim Crop].[Crop Type].Members ON ROWS
FROM [Urban Farming DW]
```

### 6.3 Production Performance Score
```mdx
WITH 
    MEMBER [Measures].[Performance Score] AS 
        ([Measures].[Yield Kg Per Hectare] * [Measures].[Water Efficiency]) / 100
SELECT 
    {[Measures].[Yield Kg Per Hectare], [Measures].[Water Efficiency], [Measures].[Performance Score]} ON COLUMNS,
    [Dim Crop].[Crop Type].Members ON ROWS
FROM [Urban Farming DW]
```

---

## 7. Ranking Queries

### 7.1 Rank Crops by Yield
```mdx
WITH 
    MEMBER [Measures].[Yield Rank] AS 
        RANK([Dim Crop].[Crop Type].CurrentMember, 
             [Dim Crop].[Crop Type].Members, 
             [Measures].[Yield Kg Per Hectare])
SELECT 
    {[Measures].[Yield Kg Per Hectare], [Measures].[Yield Rank]} ON COLUMNS,
    ORDER(
        [Dim Crop].[Crop Type].Members,
        [Measures].[Yield Kg Per Hectare],
        BDESC
    ) ON ROWS
FROM [Urban Farming DW]
```

### 7.2 Rank Regions by Water Efficiency
```mdx
WITH 
    MEMBER [Measures].[Efficiency Rank] AS 
        RANK([Dim Region].[Region Name].CurrentMember, 
             [Dim Region].[Region Name].Members, 
             [Measures].[Water Efficiency])
SELECT 
    {[Measures].[Water Efficiency], [Measures].[Efficiency Rank]} ON COLUMNS,
    ORDER(
        [Dim Region].[Region Name].Members,
        [Measures].[Water Efficiency],
        BDESC
    ) ON ROWS
FROM [Urban Farming DW]
```

---

## 8. Aggregation Queries

### 8.1 Yield by Country
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    [Dim Region].[Country].Members ON ROWS
FROM [Urban Farming DW]
```

### 8.2 Average Growing Conditions by Irrigation Type
```mdx
WITH 
    MEMBER [Measures].[Avg Growing Score] AS 
        AVG([Dim Crop].[Crop Type].Members, [Measures].[Growing Conditions Score])
SELECT 
    [Measures].[Avg Growing Score] ON COLUMNS,
    [Dim Irrigation].[Irrigation Type].Members ON ROWS
FROM [Urban Farming DW]
```

### 8.3 Total Yield by Disease Status
```mdx
SELECT 
    {[Measures].[Yield Kg Per Hectare], [Measures].[Fact Production Count]} ON COLUMNS,
    [Dim Disease].[Disease Status].Members ON ROWS
FROM [Urban Farming DW]
```

---

## 9. Comparative Analysis

### 9.1 Sowing vs Harvest Season Comparison
```mdx
SELECT 
    [Measures].[Yield Kg Per Hectare] ON COLUMNS,
    NON EMPTY CROSSJOIN(
        [Sowing Time].[Season].Members,
        [Harvest Time].[Season].Members
    ) ON ROWS
FROM [Urban Farming DW]
```

### 9.2 Healthy vs Diseased Crop Yield Comparison
```mdx
WITH 
    MEMBER [Measures].[Healthy Yield] AS 
        ([Measures].[Yield Kg Per Hectare], [Dim Disease].[Disease Status].&[None])
    MEMBER [Measures].[Diseased Yield] AS 
        ([Measures].[Yield Kg Per Hectare], [Dim Disease].[Disease Status].&[Infected])
    MEMBER [Measures].[Yield Difference] AS 
        [Measures].[Healthy Yield] - [Measures].[Diseased Yield]
SELECT 
    {[Measures].[Healthy Yield], [Measures].[Diseased Yield], [Measures].[Yield Difference]} ON COLUMNS,
    [Dim Crop].[Crop Type].Members ON ROWS
FROM [Urban Farming DW]
```

---

## 10. Advanced Queries

### 10.1 Percentage Contribution of Each Crop to Total Yield
```mdx
WITH 
    MEMBER [Measures].[Yield Percentage] AS 
        IIF(
            ([Measures].[Yield Kg Per Hectare], [Dim Crop].[Crop Type].[All]) = 0,
            0,
            ([Measures].[Yield Kg Per Hectare] / 
             ([Measures].[Yield Kg Per Hectare], [Dim Crop].[Crop Type].[All])) * 100
        ),
        FORMAT_STRING = "0.00%"
SELECT 
    {[Measures].[Yield Kg Per Hectare], [Measures].[Yield Percentage]} ON COLUMNS,
    [Dim Crop].[Crop Type].Members ON ROWS
FROM [Urban Farming DW]
```

### 10.2 Optimal Irrigation Type per Crop (Max Water Efficiency)
```mdx
SELECT 
    [Measures].[Water Efficiency] ON COLUMNS,
    GENERATE(
        [Dim Crop].[Crop Type].Members,
        TOPCOUNT(
            [Dim Irrigation].[Irrigation Type].Members,
            1,
            [Measures].[Water Efficiency]
        )
    ) ON ROWS
FROM [Urban Farming DW]
```

### 10.3 Multi-Dimension Performance Dashboard
```mdx
SELECT 
    {
        [Measures].[Yield Kg Per Hectare],
        [Measures].[Water Efficiency],
        [Measures].[Growing Conditions Score],
        [Measures].[Fact Production Count]
    } ON COLUMNS,
    NON EMPTY CROSSJOIN(
        [Dim Region].[Country].Members,
        [Dim Crop].[Crop Type].Members
    ) ON ROWS
FROM [Urban Farming DW]
```

---

## Notes

- **Cube Name**: `[Urban Farming DW]`
- **Measure Group**: `Fact Production`
- **Dimensions**:
  - `[Dim Crop]` - Crop Type
  - `[Dim Region]` - Region Name, Country
  - `[Dim Irrigation]` - Irrigation Type
  - `[Dim Disease]` - Disease Status
  - `[Harvest Time]` - Quarter, Month Name, Season
  - `[Sowing Time]` - Quarter, Month Name, Season

> **Tip**: Adjust member names (e.g., `&[Tomato]`, `&[Drip]`) based on your actual data values.
