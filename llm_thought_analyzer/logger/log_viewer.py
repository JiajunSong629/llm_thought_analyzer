from flask import Flask, render_template_string
from llm_thought_analyzer.logger import LLMLogger

app = Flask(__name__)
llm_logger = LLMLogger()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Call Logs</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/datatables@1.10.18/media/css/jquery.dataTables.min.css" rel="stylesheet">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        .table-container {
            margin: 20px;
            width: 98vw;
            max-width: none;
        }
        .container {
            max-width: none;
            width: 100%;
            padding: 0 20px;
        }
        .prompt-cell {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
        }
        .modal-content {
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal-body pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .dataTables_wrapper {
            padding: 20px;
            width: 100%;
        }
        .dataTables_filter {
            margin-bottom: 10px;
        }
        .table th {
            position: relative;
            cursor: pointer;
        }
        .table th.sorting:after,
        .table th.sorting_asc:after,
        .table th.sorting_desc:after {
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
        }
        .table th.sorting:after {
            content: "⇅";
            opacity: 0.3;
        }
        .table th.sorting_asc:after {
            content: "↑";
        }
        .table th.sorting_desc:after {
            content: "↓";
        }
        /* Resizable columns styles */
        .resizable-handle {
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 5px;
            cursor: col-resize;
            background: transparent;
        }
        .resizable-handle:hover {
            background: rgba(0, 0, 0, 0.1);
        }
        /* Resizable rows styles */
        .resizable-row-handle {
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 5px;
            cursor: row-resize;
            background: transparent;
        }
        .resizable-row-handle:hover {
            background: rgba(0, 0, 0, 0.1);
        }
        .table td, .table th {
            position: relative;
        }
        /* Ensure table takes full width */
        .table {
            width: 100% !important;
        }
        /* Make DataTables scroll horizontally */
        .dataTables_scroll {
            width: 100%;
        }
        .dataTables_scrollBody {
            width: 100% !important;
        }
    </style>
</head>
<body>
    <div class="container table-container">
        <h1>LLM Call Logs</h1>
        <div class="table-responsive">
            <table id="logsTable" class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Model Provider</th>
                        <th>Model Name</th>
                        <th>System Prompt</th>
                        <th>User Prompt</th>
                        <th>Response</th>
                        <th>Input Tokens</th>
                        <th>Output Tokens</th>
                        <th>Duration (ms)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for log in logs %}
                    <tr>
                        <td>{{ log.timestamp }}</td>
                        <td>{{ log.model_provider }}</td>
                        <td>{{ log.model_name }}</td>
                        <td class="prompt-cell" data-bs-toggle="modal" data-bs-target="#contentModal" 
                            data-content="{{ log.system_prompt }}" data-title="System Prompt">
                            {{ log.system_prompt }}
                        </td>
                        <td class="prompt-cell" data-bs-toggle="modal" data-bs-target="#contentModal" 
                            data-content="{{ log.user_prompt }}" data-title="User Prompt">
                            {{ log.user_prompt }}
                        </td>
                        <td class="prompt-cell" data-bs-toggle="modal" data-bs-target="#contentModal" 
                            data-content="{{ log.response }}" data-title="Response">
                            {{ log.response }}
                        </td>
                        <td>{{ log.input_tokens }}</td>
                        <td>{{ log.output_tokens }}</td>
                        <td>{{ log.duration_ms }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Content Modal -->
    <div class="modal fade" id="contentModal" tabindex="-1">
        <div class="modal-dialog modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="modalTitle">Content</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <pre id="modalContent"></pre>
                </div>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/fixedheader/3.2.4/js/dataTables.fixedHeader.min.js"></script>
    <script>
        $(document).ready(function() {
            var table = $('#logsTable').DataTable({
                pageLength: 25,
                order: [[0, 'desc']], // Sort by timestamp descending by default
                columnDefs: [
                    { targets: [3, 4, 5], orderable: false } // Disable sorting for prompt columns
                ],
                scrollX: true,
                fixedHeader: true,
                autoWidth: false,
                scrollCollapse: true
            });

            // Handle modal content
            $('#contentModal').on('show.bs.modal', function(event) {
                var button = $(event.relatedTarget);
                var content = button.data('content');
                var title = button.data('title');
                var modal = $(this);
                modal.find('.modal-title').text(title);
                modal.find('#modalContent').text(content);
            });

            // Add tooltips
            $('[data-bs-toggle="tooltip"]').tooltip();
        });
    </script>
</body>
</html>
"""


@app.route("/")
def view_logs():
    logs = llm_logger.get_all_logs()
    return render_template_string(HTML_TEMPLATE, logs=logs)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
