CREATE table Flower(
    flower_id int auto_increment,
    sepal_len DECIMAL(4, 2) not null,
    sepal_width DECIMAL(4, 2) not null,
    petal_len DECIMAL(4, 2) not null,
    petal_width DECIMAL(4, 2) not null,
    class varchar(60) not null,
    primary key(flower_id)
);