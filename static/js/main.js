console.log("Main JS File Loading...");
var isCall = false;
var oldVal = "";
var preVal = "";
var isbackspace = false;

var isMicEnabled = false;

function setExcelData(jsonResponse){
	var header = "<div class='thead'>";
	var keys = Object.keys(jsonResponse);
	var array = [];
	for(var i=0;i<keys.length;i++){
		header = header+"<div class='column1 header'>"+keys[i]+"</div>";
		var data = jsonResponse[keys[i]];
		var keys1 = Object.keys(data);
		for(var j=0;j<keys1.length;j++){
			if(typeof array[j] === 'undefined'){
				array[j] = "";
			}
			array[j] = array[j] + "<div class='column1'>"+data[j]+"</div>";
		}
	}
	var excel = header + "</div>";
	for(var i=0;i<array.length;i++){
		excel = excel + "<div>"+array[i]+"</div>";
	}
	document.getElementById("excel").innerHTML = excel;
}
function getExcel(){
	var url = "/excelData";
	fetch(url, 
	{
		method: 'GET',
		headers: {
			'Content-type': 'application/json',
			'Accept': 'application/json'
		}	
	})
	.then(res=>{
		if(res.ok){
			return res.json();
		}else{
			alert("something is wrong");
		}
	})
	.then(jsonResponse=>{setExcelData(jsonResponse);})
	.catch((err) => console.error(err));
}
async function getImage(){
    var url = "/image"
    var text_data = ""
    await fetch(url, 
    {
        method: 'GET',
        headers: {
            'Content-type': 'application/json',
            'Accept': 'application/json'
        }    
    })
    .then(res=>res.json())
    .then(async (data)=> await (text_data = data.text));
    document.getElementById("pText").value = document.getElementById("pText").value.trim()+" "+ text_data;
    document.getElementById("pText").focus();
    getTextBoxValue();
}

//Function to retrieve speech as text
async function getSpeechToText(){
	var url="/speech";
	await fetch(url, 
	{
		method: 'GET',
		headers: {
			'Content-type': 'application/json',
			'Accept': 'application/json'
		}	
	})
	.then(res=>{
		if(res.ok){
			return res.json();
		}else{
			alert("something is wrong");
		}
	})
	.then(async data=>
		{
		if(data.wordSaid != null){
			await setSpeechValue(data.wordSaid)
		}
	})
	.catch((err) => console.error(err));
}

async function put(url, data) {
 
    // Awaiting fetch which contains method,
    // headers and content-type and body
    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'Content-type': 'application/json'
      },
      body: JSON.stringify(data)
    });
     
    // Awaiting response.json()
    const resData = await response.json();
 
    // Return response data
    return resData;
}

function sendData(value, isClicked){
	var url = "/excelData";
	const data = {word: value, isPredict: isClicked}
	put(url,data)
	// Resolving promise for response data
	.then(jsonResponse => {setExcelData(jsonResponse);})
	// Resolving promise for error
	.catch(err => console.log(err));
}

function reset()
{
	isCall = false;
	oldVal = "";
	preVal = "";
}

function getTextBoxValue(){
	var curVal = document.getElementById("pText").value.trim();
	preVal = document.getElementById("pText").value;
	callAPI(curVal);
}

//Function to update current word, update excel and show in textbox
function setValue(obj){
	var value = obj.innerHTML.trim();
    document.getElementById("pText").value = document.getElementById("pText").value.trim()+" "+value;
	document.getElementById("pText").focus();
	sendData(value,'Y')
	getTextBoxValue();
}

function setSpeechValue(wordSaid){
	var isExisting = 'N';
	for(var i =0 ; i<3;i++){
		var obj = document.getElementById("rc"+i);
		if(obj) {
			var value = obj.innerHTML;
			if(value == wordSaid){
				isExisting = 'Y';
				break;
			}
		}
	}
	document.getElementById("pText").value = document.getElementById("pText").value.trim()+" "+wordSaid;
	document.getElementById("pText").focus();
	sendData(wordSaid, isExisting);
	getTextBoxValue();
}

async function callAPI(data){
	var url = "/prediction?data="+data;
	await fetch(url, 
	{
		method: 'GET',
		headers: {
			'Content-type': 'application/json',
			'Accept': 'application/json'
		}
	}
	).then(res=>{
		if(res.ok){
			return res.json();
		}else{
			alert("something is wrong");
		}
	}
	).then(jsonResponse=>{
		let words = jsonResponse;
		if(words.length>0) {
			var acuracy_zero = 0;
			for(let i = 0; i < words.length; i++) {
				let obj = words[i];
				var accuracy = obj.accuracy;
				if(accuracy > 0) {
					//var col = 360 * accuracy / 100;
					document.getElementById("rc"+i).innerHTML = obj.name;
					document.getElementById("ac"+i).innerHTML = accuracy+"%";
					document.getElementById("rc"+i).style.display = "block";
					document.getElementById("ac"+i).style.display = "block";
				} else {
					acuracy_zero++;
					document.getElementById("rc"+i).style.display = "none";
					document.getElementById("ac"+i).style.display = "none";
				}
			}
			if(acuracy_zero==3){
				document.getElementById("predict").style.display = "none";
				document.getElementById("nopredict").style.display = "block";
			} else {
				document.getElementById("predict").style.display = "block";
				document.getElementById("nopredict").style.display = "none";
			}
			} else {
				document.getElementById("predict").style.display = "none";
				document.getElementById("nopredict").style.display = "block";
			}
			// Check if the mic is enabled/disabled
			if(isMicEnabled){
				getSpeechToText();
			}
	} 
	).catch((err) => console.error(err));
}

function processData(event) {
	var sentence = "";
	var curVal = document.getElementById("pText").value.trim();
	if(curVal==""){
		document.getElementById("pText").value = curVal;
		document.getElementById("predict").style.display = "none";
		document.getElementById("nopredict").style.display = "none";
		reset();
		return;
	}
	if(curVal.indexOf(".")>0){
		var sentences = curVal.split(".");
		sentence = sentences[sentences.length-1]
	} else {
		sentence = curVal;
	}
	if(oldVal != curVal || isbackspace) {
		if(event.keyCode == 32){
			if(!isCall && curVal!="") {
				if(preVal != curVal){
					var words = curVal.split(" ");
					var val = words[words.length-1].trim();
					sendData(val,'N');
				}
				callAPI(sentence.trim());
				isCall = true;
				oldVal = curVal;
				preVal = curVal;
				isbackspace = false;
			}
		} else if(event.keyCode == 190){
			isCall = true;
			oldVal = curVal;
			preVal = curVal;
			document.getElementById("predict").style.display = "none";
		} else if(event.keyCode == 8){
			oldVal = curVal;
			preVal = curVal;
			isbackspace = true;
			if(curVal != document.getElementById("pText").value){
				callAPI(sentence.trim());
				isCall = true;
			} else {
				isCall = false;
			}
		} else {
			isCall = false;
		}
	} else {
		isCall = true;
	}
}
