var express = require('express')
var router = express.Router()

var multer = require('multer')
var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname)
    }
  })


var upload = multer({ storage: storage })

var spawn = require('child_process').spawn


module.exports = router


router.post('/', upload.single('uploaded_file'), (req, res) => {

    var path = req.file.path
    // var path = "uploads/16.jpg"

    var process = spawn('python3', ['python/test.py', path])

    process.stdout.on('data', (data) => {
        data = String(data)
        // console.log(data)

        json = JSON.parse(data)

        res.json({status:200, result: json})
    })

    process.stderr.on('data', (data) => {
        data = String(data)
        // console.log(data)

    })

    process.stdout.on('error', (error) => {
        res.json({status:500})
    })
})
