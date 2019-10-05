var express = require('express')
var router = express.Router()

var name = require('./name/index')
var shape = require('./shape/index')
var photo = require('./photo/index')

module.exports = router

router.use('/name', name)
router.use('/shape', shape)
router.use('/photo', photo)

router.get('/', (req, res) => {
    console.log('main page open')
    res.render('api.ejs')
})