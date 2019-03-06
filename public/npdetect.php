<?php
    function detectNumberplate($filename) {
        $data = array("doRecognizing" => 1, "includeReestrInfo" => 1, "image_url" => "https://nomeroff.net.ua/uploads/".$filename);

        $data_string = json_encode($data);

        $ch = curl_init('http://dev.it.ria.com:8901/detect');
        curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "POST");
        curl_setopt($ch, CURLOPT_POSTFIELDS, $data_string);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_HTTPHEADER, array(
            'Content-Type: application/json',
            'Content-Length: ' . strlen($data_string))
        );
        $result = curl_exec($ch);
        return json_decode($result);
    }

    function getRealIpAddr() {
        if (!empty($_SERVER['HTTP_CLIENT_IP'])) {   //check ip from share internet
            $ip=$_SERVER['HTTP_CLIENT_IP'];
        }  elseif (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {  //to check ip is pass from proxy
            $ip=$_SERVER['HTTP_X_FORWARDED_FOR'];
        } else {
            $ip=$_SERVER['REMOTE_ADDR'];
        }
        return $ip;
    }

    function checkReCaptcha($token) {
        $data = array("secret" => "6LdvT4sUAAAAABeNHOHJXaj-UE_CnpRRHkJBKGcB", "response" => $token); //, remoteip => getRealIpAddr()

        $ch = curl_init('https://www.google.com/recaptcha/api/siteverify');
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
        $result = curl_exec($ch);
        curl_close($ch);
        return json_decode($result);
    }

    // Init variables
    $version = '1.7';
    $reCaptcha = checkReCaptcha($_POST["token"]);
    //$reCaptcha = [ 'success' => false ];

    if ( !$reCaptcha->success ) {
    //if ( false ) {
        $data = [ 'error_id' => 4, success=>false, 'error_message' => 'reCapcha error: '.implode(",",$reCaptcha->{'error-codes'}) ]; 
    } elseif ( 0 < $_FILES['file']['error'] ) {
        $data = [ 'error_id' => 1, success=>false, 'error_message' => $_FILES['file']['error'] ];
    } else {
        $file = $_FILES['file']['tmp_name'];
        if (file_exists($file))
        {
            $hash = md5_file($file).".jpeg";

            $imagesizedata = getimagesize($file);
            if ($imagesizedata === FALSE) {
                //not image
                $data = [ 'error_id' => 2, success=>false, 'error_message' => 'Uploaded file "'.$_FILES['file']['name'].'" is not picture' ];
            } else {
                //image
                //use $imagesizedata to get extra info
                $target_file='uploads/' . $hash;
                move_uploaded_file($file, $target_file);
                $numberplateData = detectNumberplate($hash);
                $data = [ 'error_id' => 0, success=>true, 'imagesizedata' => $imagesizedata, 'numberplateData' => $numberplateData ];
            }
        } else {
            //not file
            $data = [ 'error_id' => 3, success=>false, 'error_message' => 'Error file uploading' ];
        }
    }
    header('Content-Type: application/json');
    $data['reCaptcha'] = $reCaptcha;
    $data['version'] = $version;
    echo json_encode($data);
?>