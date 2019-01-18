<?php
    function detectNumberplate($filename) {
        $data = array("doRecognizing" => 1, "image_url" => "https://nomeroff.net.ua/uploads/".$filename);

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


    if ( 0 < $_FILES['file']['error'] ) {
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
    echo json_encode($data);
?>