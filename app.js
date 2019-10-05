var express = require('express')
var app = express()
var bodyParser = require('body-parser')

var router = require('./router/index')

app.listen(8000, () => {
    console.log("Start Server with port", 8000)
})


app.use(bodyParser.json({limit:'100mb', extended: true}))
app.use(bodyParser.urlencoded({limit:'100mb', extended: true}))
app.set('view engine', 'ejs')

app.use(router)
