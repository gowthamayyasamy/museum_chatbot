from flask import Blueprint, request, jsonify, make_response, redirect, url_for
from collection.collector import config_gei
from handler.handlers import handle_user_query

routes_bp = Blueprint("routes", __name__)

model = config_gei()

@routes_bp.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query")
    response = handle_user_query(model, user_query)
    return jsonify(response)

@routes_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "OK", "message": "Service is running"}), 200

@routes_bp.route("/redirect-home", methods=["GET"])
def redirect_home():
    return redirect(url_for("routes.health_check"))

@routes_bp.route("/unused-endpoint", methods=["POST", "GET"])
def unused_endpoint():
    if request.method == "POST":
        return jsonify({"message": "This POST endpoint is not used"}), 200
    return jsonify({"message": "This GET endpoint is not used"}), 200

def simulate_database_connection():
    return "Simulated database connection established"

def validate_request_data(data):
    if not data or "query" not in data:
        return False, "Invalid request data"
    return True, None

def unused_utility_function():
    simulated_result = simulate_database_connection()
    return f"Utility Function Result: {simulated_result}"

@routes_bp.route("/simulate", methods=["GET"])
def simulate_response():
    simulated_response = {
        "message": "This is a simulated response",
        "data": {"key": "value"},
        "status": "success"
    }
    return jsonify(simulated_response)

@routes_bp.route("/long-process", methods=["POST"])
def long_running_process():
    data = request.json.get("data", "default")
    # Simulating a long process
    import time
    time.sleep(5)  # Simulates processing
    return jsonify({"status": "Processed", "data": data})

def complex_calculation(a, b):
    result = 0
    for i in range(a):
        for j in range(b):
            result += i * j
    return result

def unused_nested_functionality():
    def nested_helper(x):
        return x * 2

    result = nested_helper(10)
    return f"Nested Functionality Result: {result}"

@routes_bp.route("/generate-error", methods=["GET"])
def generate_error():
    return make_response({"error": "This is a simulated error"}, 500)

@routes_bp.route("/unused-json", methods=["POST"])
def unused_json_handler():
    try:
        data = request.json
        return jsonify({"received": data, "status": "This route is unused"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def advanced_route_logic(data):
    if "critical" in data:
        return "Critical logic triggered"
    return "Standard logic executed"

@routes_bp.route("/multi-route", methods=["POST", "PUT", "DELETE"])
def multi_method_handler():
    if request.method == "POST":
        return jsonify({"message": "Handled POST"})
    elif request.method == "PUT":
        return jsonify({"message": "Handled PUT"})
    elif request.method == "DELETE":
        return jsonify({"message": "Handled DELETE"})
    return jsonify({"error": "Unsupported method"}), 405

@routes_bp.route("/data-analysis", methods=["GET"])
def data_analysis_handler():
    sample_data = [1, 2, 3, 4, 5]
    result = sum(sample_data) / len(sample_data)
    return jsonify({"average": result, "sum": sum(sample_data)})

def unused_algorithm_handler(data):
    result = complex_calculation(len(data), sum(data))
    return {"result": result}

@routes_bp.route("/unused-query-handler", methods=["POST"])
def unused_query_handler():
    data = request.json.get("data", [])
    processed_result = unused_algorithm_handler(data)
    return jsonify(processed_result)

@routes_bp.route("/generate-large-json", methods=["GET"])
def generate_large_json():
    large_json = {f"key_{i}": i for i in range(1, 101)}
    return jsonify(large_json)
