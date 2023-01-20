# Employee Test DB
From [MySQL Docs](https://dev.mysql.com/doc/employee/en/employees-installation.html)

1. Pull a MariaDB Server Image `docker run -d --rm --name mariadbcont -e MYSQL_ROOT_PASSWORD=root -p 3306:3306 mariadb`
2. Verify image `docker images`
3. Run container in bash mode `docker exec -it mariadbcont bash`
4. Install git:
    `apt-get update`
    `apt-get install git`
5. Download the Employee DB from [GitHub](https://github.com/datacharmer/test_db)  `git clone https://github.com/datacharmer/test_db.git`
6. Import data into DB `mysql < employees.sql -u root -proot`
7. In SQL:
   1. `show databases;`
   2. `use employees;`
   3. `show tables;`
   4. `show columns from employees;`
   5. `show columns from salaries;`
   6. `show columns from titles;`
   7. Given the title of the people and its salary, How much could be your salary based on having title in 2023?
      1. We create a query for selecting the salary, title and date from tables **salaries** and **titles**
        <pre>
        SELECT salaries.salary, titles.title, salaries.from_date
        FROM salaries
        INNER JOIN titles ON salaries.emp_no=titles.emp_no
        limit 10;</pre>
      2. Then, we select distinct values of title in table titles
        <pre>
        select distinct title
        from titles;</pre>
      3. We will choose *Senior Engineer* for our task.
      4. Our final query would be
        <pre>
        SELECT titles.title, salaries.salary, salaries.from_date
        FROM titles
        INNER JOIN salaries ON titles.emp_no=salaries.emp_no
        WHERE titles.title="Senior Engineer"
        limit 10;</pre>
       5. Run the notebook `main.ipynb`
    



