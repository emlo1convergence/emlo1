function loadImage(upload_event) {
    const files = upload_event.target.files;

    if (files.length === 0) {
      return;
    }

    const mimeType = files[0].type;
    if (mimeType.match(/image\/*/) == null) {
      return;
    }
    
    var name_of_image = files[0].name;
    var img_name_tag = document.getElementById("img_name");
    var inp_img_container = document.getElementById("inp_img_container");
    var inp_img_tag = document.getElementById("inp_img_tag");
    var submit_btn = document.getElementById("pred_btn");

    var input_image_details = document.getElementById("input_image_details");

    var img_url = URL.createObjectURL(files[0]);
    var img_name = name_of_image;

    inp_img_container.style.display = "block";
    inp_img_tag.src = img_url;

    img_name_tag.innerHTML = img_name;
    submit_btn.disabled = false;

    input_image_details.innerHTML = "Image uploaded by the user, " + img_name;
    input_image_details.style.display = "block";
}

function closeImage() {
    var img_input = document.getElementById("img_upload");
    var img_name_tag = document.getElementById("img_name");
    var inp_img_tag = document.getElementById("inp_img_tag");
    var inp_img_container = document.getElementById("inp_img_container");

    // collecting the tags containing image details
    var input_image_details = document.getElementById("input_image_details");

    img_name = "No Image Selected";
    img_url = "";
    img_input.value = "";

    try {

        var results = document.getElementById("results");
        results.parentNode.removeChild(results);
    }
    catch { }

    inp_img_tag.src = img_url;
    inp_img_container.style.display = "none";
    input_image_details.style.display = "none";
    img_name_tag.innerHTML = img_name;
}