import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/main.dart';

void main() {
  testWidgets('App shows navigation destinations', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    expect(find.text('识别'), findsOneWidget);
    expect(find.text('提交'), findsOneWidget);
  });
}
