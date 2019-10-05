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


router.get('/', (req, res) => {
    var character1 = req.query.character1
    var character2 = req.query.character2
    var color = req.query.color
    var shape = req.query.shape
    var page = req.query.page

    if (!character1 && character2) {
        character1 = character2
        character2 = undefined
    }

    if (!character1) character1 = "%%"
    if (!character2) character2 = "%%"
    if (!color) color = "%%"
    if (!shape) shape = "%%"
    if (!page) page = 1

    var sql = "SELECT * FROM PILLS WHERE CHARACTER1 LIKE ? AND (CHARACTER2 LIKE ? OR CHARACTER2 IS NULL) AND COLOR LIKE ? AND SHAPE LIKE ? LIMIT ?, 20"

    var query = connection.query(sql,[character1, character2, color, shape, (page - 1) * 20], (err, rows) => {
        if (err) {
            res.json({status: 500})
            throw err
        }
        res.json({status:200, 'result':rows})
    })

    
    
})


router.post('/', (req, res) => {
    var body = req.body
    var pill_main = body.pill_main
    var name = pill_main.name
    var character1 =  pill_main.character1
    var character2 = pill_main.character2
    var color = pill_main.color
    var shape = pill_main.shape
    var img = pill_main.img

    var pill_details = body.pill_details

    var basic = pill_details.basic

    var main = basic.main
    var shapes = basic.shapes
    var element = basic.element
    var store = basic.store

    var effect = pill_details.effect
    var howto = pill_details.howto
    var caution = pill_details.caution

    success_result = {status : 200}
    failure_result = {status : 300}


    var data = {name, character1 ,character2, color, shape, img, main, shapes, element, store, effect, howto, caution}
    var query = connection.query('SELECT NAME FROM PILLS WHERE NAME = ?',[name], (err, rows) => {
        if (err) {
            res.json(failure_result)
            throw err
        }
        if (rows.length == 0) {
            var query2 = connection.query('INSERT INTO PILLS SET ?', data, (err, rows) => {
                if (err) {
                    res.json(failure_result)
                    throw err
                }
                res.json(success_result)
                console.log('insert into db name =>', name)
            })
        } else {
            console.log('same name is already in db =>', name)
            res.json(failure_result)
        }
    })

})