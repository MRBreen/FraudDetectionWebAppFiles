
let get_input_data = function() {
    let object_id = $("input#object_id").val()
    return {'object_id': parseInt(object_id),
           }
};

let send_data_json = function(coefficients) {
    $.ajax({
        url: '/predict',
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_solutions(data);
        },
        data: JSON.stringify(coefficients)
    });
};

let display_solutions = function(solutions) {
    $("span#predict").html(solutions)
};


$(document).ready(function() {

    $("button#predict").click(function() {
        let coefficients = get_input_data();
        send_data_json(coefficients);
    })

})
