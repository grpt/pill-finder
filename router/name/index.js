var express = require('express')
var router = express.Router()
var mysql = require('mysql')

var connection = mysql.createConnection({
    host : 'localhost',
    port : 3306,
    user : 'root',
    password : 'apple',
    database : 'swgong'
})

connection.connect()

module.exports = router


router.get('/:name', (req, res) => {
    var name = req.params.name
    name = '%' + name + '%'
    var page = req.query.page
    if (!page) page = 1
    var sql = 'select * from PILLS Where name LIKE ? LIMIT ?, 20'
    var query = connection.query(sql, [name, (page - 1) * 20], (err, rows) => {
        if (err) {
            res.json({status: 500})
            throw err
        }
        res.json({status:200, 'result':rows})
    })
})
